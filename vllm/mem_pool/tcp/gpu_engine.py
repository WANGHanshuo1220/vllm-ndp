import psutil
import torch
from typing import List, Dict, TypeAlias, Tuple
import asyncio
import time
import traceback

from vllm.mem_pool.attention import Attention
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.worker.worker import CacheEngine
from vllm.executor.cpu_executor import _verify_and_get_model_config
from vllm.sequence import Sequence, SequenceGroup
from vllm.inputs.data import LLMInputs
from vllm.utils import init_logger
from vllm.core.interfaces import AllocStatus

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig, MemPoolConfig, EngineConfig)
from vllm.model_executor.model_loader.utils import (set_default_torch_dtype)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from vllm.mem_pool.util import (KVRequestTracker, KVTransferData, 
                                StoreKVRequest, AttentionComputation)
from vllm.mem_pool.radix_tree_cache import RadixCache
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
import threading
import concurrent.futures as cf

logger = init_logger(__name__)

class Memory_pool_engine():

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        mem_pool_config: MemPoolConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        max_kv_workers=6
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.mem_pool_config = mem_pool_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.max_kv_workers = max_kv_workers

        self.dimension = (self.cache_config.block_size *
                          self.model_config.get_num_attention_heads(
                              self.parallel_config) *
                          self.model_config.get_head_size())
        self.dtype = self.model_config.dtype

        self.attention_unit = self._create_attention()
        self.cache_enigne = self._create_cache_engine()
        self.block_manager = self._create_block_manager()

        self.radix_tree = RadixCache(block_size=self.cache_config.block_size)

        self.kv_transfer_running = False
        self.kv_transfer_request_tracker: KVRequestTracker = None
        self.executor = cf.ThreadPoolExecutor(max_workers=self.max_kv_workers)
        self.block_manager_lock = threading.Lock()

        # self.attention_running_loop = False
        # self.attention_request_tracke:
        torch.set_default_dtype(torch.float16)
    
    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown()

    def _create_attention(self) -> Attention:
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        head_size = self.model_config.get_head_size()
        scale = 1.0

        target_device = torch.device(self.device_config.device_type)
        with set_default_torch_dtype(self.model_config.dtype):
            with target_device:
                attn = Attention(
                    num_heads,
                    head_size,
                    scale=scale,
                    cache_config=self.cache_config,
                    mem_pool_config=self.mem_pool_config
                ).to(target_device)
        return attn.eval()

    def _create_cache_engine(self) -> CacheEngine:
        cache_cpu_block_size = CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config)

        num_gpu_blocks = 40000
        num_cpu_blocks = 0

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        logger.info(f"Total number of gpu blocks = {num_gpu_blocks}")
        logger.info(f"Total number of cpu blocks = {num_cpu_blocks}")

        return CacheEngine(self.cache_config, self.model_config,
                              self.parallel_config, self.device_config)

    def _create_block_manager(self) -> BlockSpaceManagerV2:
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= self.parallel_config.pipeline_parallel_size

        num_cpu_blocks = self.cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= self.parallel_config.pipeline_parallel_size

        return BlockSpaceManagerV2(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=True)

    def _get_available_cpu_memory(self):
        memory_info = psutil.virtual_memory()
        return memory_info.available * \
            self.cache_config.gpu_memory_utilization

    @classmethod
    def create(cls, engine_config: EngineConfig) -> "Memory_pool_engine":
        model_config: ModelConfig = engine_config.model_config
        cache_config: CacheConfig = engine_config.cache_config
        mem_pool_config: MemPoolConfig = engine_config.mem_pool_config
        parallel_config: ParallelConfig = engine_config.parallel_config
        device_config: DeviceConfig = engine_config.device_config

        engine = cls(
            model_config,
            cache_config,
            mem_pool_config,
            parallel_config,
            device_config,
        )

        return engine

    """ Below are class network interfaces """

    def _can_allocate(self, seq_group) -> bool:
        return self.block_manager.can_allocate(seq_group)

    def _allocate(self) -> None:
        pass

    def _update_radix_tree(
        self,
        token_ids: List[int],
        block_ids: List[int]
    ) -> None:
        self.radix_tree.insert(token_ids, block_ids)

    def _kv_transfer_loop(self):
        while True:
            new_requests: List[KVTransferData] = \
                self.kv_transfer_request_tracker.get_new_requests()
            
            assert len(new_requests) <= self.max_kv_workers-1, \
                f"Don't have enough workers{self.max_kv_workers-1} to serve"
                
            futures = [self.executor.submit(
                self.store_kv, seq.seq_id, seq.token_ids,
                seq.blocks_to_tensor) for seq in new_requests]
            
            cf.wait(futures)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread: {e}")
                    traceback.print_exc()

    def _start_kv_transfer_thread(self) -> None:

        self.kv_transfer_request_tracker = KVRequestTracker(
            self.max_kv_workers - 1)

        self.executor.submit(self._kv_transfer_loop)

        self.kv_transfer_running = True

        logger.info("kv transfer thread started")

    def _preprocess_request(
        self, 
        request: StoreKVRequest
    ) -> Tuple[int, List[int], Dict[int, List[torch.tensor]]]:
        # NOTE: Restore original tensor
        blocks_to_tensor = {}
        for block_id, tensor_list in request.tensor_data.items():
            block_to_tensor = []
            for layer_tensor_list in tensor_list:
                org_layer_tensor = torch.tensor(layer_tensor_list)
                block_to_tensor.append(org_layer_tensor)
            blocks_to_tensor[block_id] = block_to_tensor

        return request.seq_id, request.token_ids, blocks_to_tensor

    def add_kv_transfer_request(
        self, 
        request: StoreKVRequest
    ) -> None:
        logger.debug(f"recieve {request.seq_id} at {time.time():.4f}")

        # Start running loop if not
        if not self.kv_transfer_running:
            self._start_kv_transfer_thread()

        # Preprocessing request data
        seq_id, token_ids, blocks_to_tensor = self._preprocess_request(request)

        # Add this request to waiting queue
        data = KVTransferData(seq_id, token_ids, blocks_to_tensor)
        self.kv_transfer_request_tracker.add_request(seq_id, data)

    def store_kv(
        self, 
        seq_id: int, 
        token_ids: List[int],
        blocks_to_tensor: Dict[int, List[torch.tensor]]
    ) -> None:
        # Create a sequence group
        sequence = Sequence(
            seq_id=seq_id,
            inputs={"prompt_token_ids": token_ids},
            block_size=self.cache_config.block_size,
        )
        seq_group = SequenceGroup(
            request_id=seq_id,
            seqs=[sequence],
            arrival_time=time.time()
        )

        # NOTE: Here need lock for multi-processing safety
        with self.block_manager_lock:
            # 1. Check if we can allocate blocks for this seq
            can_allocate = self._can_allocate(seq_group)

            # 2. Allocate if we can and free some blocks if neccessary
            if can_allocate == AllocStatus.OK:
                # allocate blocks
                self.block_manager.allocate(seq_group)
            elif can_allocate == AllocStatus.LATER:
                # free some blocks and allocate
                assert False, "Later exception is not implemented yet"
            else:
                assert False, "Abort exception is not implemented yet"

        # 3. Store tensors to cpu cache
        allocated_blocks = self.block_manager.get_block_table(sequence)
        blocks_reusable = self.block_manager.get_block_reusable(sequence)
        assert len(allocated_blocks) == len(blocks_reusable)
        assert len(allocated_blocks) == len(blocks_to_tensor)
        for i, (_, kv_tensor_layers) in enumerate(blocks_to_tensor.items()):
            block_id = allocated_blocks[i]
            if not blocks_reusable[i]:
                for i in range(len(kv_tensor_layers)):
                    # tensor = kv_tensor_layers[i].view(2, self.dimension)
                    tensor = torch.tensor(kv_tensor_layers[i], device="cuda")
                    self.cache_enigne.gpu_cache[i][:,block_id,:,:,:] = tensor
                logger.debug(f"Save {block_id} of seq {seq_id} successfuly")
            else:
                logger.debug(f"Reuse {block_id} of seq {seq_id}")

        # 4. Update radix tree
        blocks = list(blocks_to_tensor.keys())
        self._update_radix_tree(token_ids, blocks)
        logger.debug(f"Store seq {seq_id} to radix tree successfully")

    def _create_cpu_attn_metadata(
        self,
        seq_lens_tensor: torch.Tensor,
        max_decode_seq_len: int,
        num_decode_tokens: int,
        seq_lens: List[int]
    ) -> FlashAttentionMetadata:
        cpu_attn_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            block_tables=None,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=None,
            seq_lens=seq_lens,
            max_query_len=None,
            max_prefill_seq_len=0,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            use_cuda_graph=False
        )
        return cpu_attn_metadata
    
    def _convert_list_to_tensor_padding(
        self, 
        _block_tables: List
    ) -> torch.Tensor:
        max_len = max(len(sublist) for sublist in _block_tables)
        
        padded_block_tables = [sublist + [0] * (max_len - len(sublist)) 
                               for sublist in _block_tables]
        
        tensor_block_tables = torch.tensor(padded_block_tables,
                                           dtype=torch.int32,
                                           device="cuda")
        
        return tensor_block_tables

    def compute_attention(self, request: AttentionComputation):

        # 1. Prepare all the data
        query = torch.tensor(request.query, dtype=self.dtype, device="cuda")
        key = torch.tensor(request.key, dtype=self.dtype, device="cuda")
        value = torch.tensor(request.value, dtype=self.dtype, device="cuda")
        
        cpu_attn_metadata = self._create_cpu_attn_metadata(
            seq_lens_tensor=torch.tensor(request.seq_len_tensor, 
                                         dtype=torch.int32, device="cuda"),
            max_decode_seq_len=request.max_decode_seq_len,
            num_decode_tokens=request.num_decode_tokens,
            seq_lens=request.seq_lens
        )

        _block_tables = []
        _slot_mapping = []
        for i, (seq_id, token_ids) in enumerate(request.seqs_data.items()):
            sequence = Sequence(
                seq_id=seq_id,
                inputs={"prompt_token_ids": token_ids},
                block_size=self.cache_config.block_size,
            )
            seq_group = SequenceGroup(
                request_id=seq_id,
                seqs=[sequence],
                arrival_time=time.time()
            )

            # 2. Allocate blocks if necessary
            while not self.block_manager.has_seq(sequence):
                continue
            else:
                assert self.block_manager.can_append_slots(
                    seq_group, num_lookahead_slots=0)
            
                self.block_manager.append_slots(
                    sequence, num_lookahead_slots=0)

        
            # 3. Get block info and slot_mapping to attn_matadata
            block_ids = self.block_manager.get_block_table(sequence)
            _block_tables.append(block_ids)

            offset = request.seq_lens[i] 
            if offset > self.cache_config.block_size:
                offset = offset % self.cache_config.block_size

            slot_mapping = (block_ids[-1] * self.cache_config.block_size
                            + offset - 1)
            _slot_mapping.append(slot_mapping)
        
        block_tables = self._convert_list_to_tensor_padding(_block_tables)
        cpu_attn_metadata.block_tables = block_tables
        cpu_attn_metadata.slot_mapping = torch.tensor(_slot_mapping,
                                                      dtype=torch.int64,
                                                      device="cuda")

        # 4. Attention computation
        layer = request.layer

        assert query.is_cuda
        assert key.is_cuda
        assert value.is_cuda
        assert self.cache_enigne.gpu_cache[layer].is_cuda
        output = self.attention_unit(query, key, value,
                                     self.cache_enigne.gpu_cache[layer],
                                     cpu_attn_metadata)
        return {"result": output.tolist()}
        