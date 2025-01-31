import psutil
import torch
from typing import List, Dict, Tuple, Optional, AsyncGenerator
import asyncio
import time

from vllm.mem_pool.attention import Attention
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.worker.cpu_worker import CPUCacheEngine
from vllm.executor.cpu_executor import _verify_and_get_model_config
from vllm.sequence import Sequence, SequenceGroup
from vllm.inputs.data import LLMInputs
from vllm.utils import init_logger
from vllm.core.interfaces import AllocStatus

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig, MemPoolConfig, EngineConfig)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from vllm.mem_pool.util import (KVRequestTracker, KVTransferData, 
                                StoreKVRequest, AttentionComputation,
                                GetKVRequest, CPU_KVCACHE_DIMENSION)
from vllm.mem_pool.radix_tree_cache import RadixCache
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata
import threading
from vllm.utils import make_async

logger = init_logger(__name__)

SEND_DELTA_TIME = 0.1

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

        torch.set_default_dtype(torch.bfloat16)
        self.attention_unit = self._create_attention()
        self.cache_enigne = self._create_cache_engine()
        self.block_manager = self._create_block_manager()

        self.radix_tree = RadixCache(block_size=self.cache_config.block_size)

        self.kv_transfer_request_tracker: KVRequestTracker = None
        self._background_kv_transfer_loop: Optional[asyncio.Task] = None

        self.send_delta = False
        self.last_cache_delta_send_time = 0.0
        self.time_lock = threading.Lock()

    def _create_attention(self) -> Attention:
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        model_config = self.model_config.hf_config
        num_kv_heads = getattr(model_config, "num_key_value_heads", 
                               model_config.num_attention_heads), 
        num_kv_heads = num_kv_heads[0]
        head_size = self.model_config.get_head_size()
        hidden_size = self.model_config.get_hidden_size()
        total_num_head = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        head_dim = hidden_size // total_num_head
        scale = head_dim**-0.5
        return Attention(
            num_heads,
            head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=self.cache_config,
            mem_pool_config=self.mem_pool_config
        )

    def _create_cache_engine(self) -> CPUCacheEngine:
        cache_cpu_block_size = CPUCacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.cache_config.cache_dtype,
            self.model_config, self.parallel_config)

        num_cpu_blocks = int(self._get_available_cpu_memory() //
                             cache_cpu_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        self.cache_config.num_gpu_blocks = num_cpu_blocks
        self.cache_config.num_cpu_blocks = 0
        logger.info(f"Total number of cpu blocks = {num_cpu_blocks}")

        _verify_and_get_model_config(self.model_config)
        return CPUCacheEngine(self.cache_config, self.model_config,
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

    def _update_radix_tree(
        self,
        token_ids: List[int],
        block_ids: List[int]
    ) -> None:
        self.radix_tree.insert(token_ids, block_ids)

    async def _kv_transfer_loop(self):
        while True:
            new_requests: List[KVTransferData] = \
                self.kv_transfer_request_tracker.get_new_requests()
            
            assert len(new_requests) <= self.max_kv_workers-1, \
                f"Don't have enough workers{self.max_kv_workers-1} to serve"
                
            if len(new_requests) > 0:
                tasks = [
                    asyncio.create_task(self.store_kv(
                        seq.seq_id, seq.token_ids,
                        seq.blocks_to_tensor, 
                        seq.to_free_seqs_list
                    )) for seq in new_requests
                ]

                await asyncio.gather(*tasks)

            await asyncio.sleep(0)

    def _start_kv_transfer_thread(self) -> None:

        self.kv_transfer_request_tracker = KVRequestTracker(
            self.max_kv_workers - 1)

        self._background_kv_transfer_loop = asyncio.get_event_loop(
        ).create_task(self._kv_transfer_loop())

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

        return (request.seq_id, request.token_ids, 
                blocks_to_tensor, request.to_free_seq_list)

    def _pass_daley(self, now: float) -> bool:
        return True
        res = False

        if self.time_lock.acquire(blocking=False):
            if not self.send_delta:
                self.send_delta = True
                self.last_cache_delta_send_time = now
                return True

            res = (now - self.last_cache_delta_send_time) > SEND_DELTA_TIME
            if res:
                self.last_cache_delta_send_time = now
        
        return res

    async def add_kv_transfer_request(
        self, 
        request: StoreKVRequest
    ) -> None:
        logger.info(f"recieve {request.seq_id} at {time.time():.4f}")

        # Start running loop if not
        if self._background_kv_transfer_loop is None:
            self._start_kv_transfer_thread()

        # Preprocessing request data
        (seq_id, token_ids, blocks_to_tensor, 
         to_free_seqs_list) = self._preprocess_request(request)

        # Add this request to waiting queue
        data = KVTransferData(seq_id, token_ids, 
                              blocks_to_tensor, to_free_seqs_list)
        self.kv_transfer_request_tracker.add_request(seq_id, data)
        
        if self._pass_daley(time.time()):
            add_delta, pop_delta = self.block_manager.get_cached_blocks_delta()
            return {"has_result": True, 
                    "add_delta": add_delta, 
                    "pop_delta": pop_delta} 
        else:
            return {"has_result": False} 

    def _store_kv_tensor(
        self,
        seq_id: str,
        allocated_blocks: List[int],
        blocks_reusable: List[bool],
        blocks_to_tensor: Dict[int, List[torch.tensor]],
    ) -> None:
        for i, (_, kv_tensor_layers) in enumerate(blocks_to_tensor.items()):
            block_id = allocated_blocks[i]
            if not blocks_reusable[i]:
                for j in range(len(kv_tensor_layers)):
                    tensor = kv_tensor_layers[j].view(2, self.dimension)
                    self.cache_enigne.cpu_cache[j][:,block_id,:] = tensor
                logger.debug(f"Save {block_id} of seq {seq_id} successfuly")
            else:
                logger.debug(f"Reuse {block_id} of seq {seq_id}")

    async def store_kv(
        self, 
        seq_id: str, 
        token_ids: List[int],
        blocks_to_tensor: Dict[int, List[torch.tensor]],
        to_free_seqs_list: List[str],
    ) -> None:
        logger.info(f"store {seq_id} at {time.time():.4f}")
        # First free blocks of finished seqs
        for to_free_seq_id in to_free_seqs_list:
            block_table = self.block_manager.block_tables[to_free_seq_id]
            to_free_token_ids = block_table._get_all_token_ids()
            to_free_sequence = Sequence(
                seq_id=to_free_seq_id,
                inputs={"prompt_token_ids": to_free_token_ids},
                block_size=self.cache_config.block_size,
            )
            self.block_manager.free(to_free_sequence)

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

        # 1. Check if we can allocate blocks for this seq
        can_allocate = self.block_manager.can_allocate(seq_group)

        # 2. Allocate if we can and free some blocks if neccessary
        if can_allocate == AllocStatus.OK:
            # allocate blocks
            self.block_manager.allocate(seq_group)
        elif can_allocate == AllocStatus.LATER:
            # Add this request to later queue
            data = KVTransferData(seq_id, token_ids, 
                                  blocks_to_tensor, to_free_seqs_list)
            self.kv_transfer_request_tracker.add_later_request(seq_id, data)
        else:
            assert False, "Abort exception is not implemented yet"

        # 3. Store tensors to cpu cache
        allocated_blocks = self.block_manager.get_block_table(sequence)
        blocks_reusable = self.block_manager.get_block_reusable(sequence)
        assert len(allocated_blocks) == len(blocks_reusable)
        assert len(allocated_blocks) == len(blocks_to_tensor)

        await make_async(self._store_kv_tensor)(
            seq_id, allocated_blocks, blocks_reusable, blocks_to_tensor
        )

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
    ) -> TorchSDPAMetadata:
        cpu_attn_metadata = TorchSDPAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            block_tables=[],
            num_decode_tokens=num_decode_tokens,
            slot_mapping=[],
            is_prompt=False,
            seq_lens=seq_lens
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
                                           dtype=torch.int32)
        
        return tensor_block_tables

    def _prefetch(self, tensors: List[torch.tensor], stride: int = 32):
        for tensor in tensors:
            for i in range(0, tensor.size(0), stride):
                tensor[i] += 0.0

    async def compute_attention(
        self, 
        request: AttentionComputation
    ) -> AsyncGenerator[Dict, None]:
        # 1. Prepare all the data
        t1 = time.time()
        query = torch.tensor(request.query, dtype=torch.bfloat16)
        key = torch.tensor(request.key, dtype=torch.bfloat16)
        value = torch.tensor(request.value, dtype=torch.bfloat16)

        cpu_attn_metadata = self._create_cpu_attn_metadata(
            seq_lens_tensor=torch.tensor(request.seq_len_tensor, 
                                         dtype=torch.int32),
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
                await asyncio.sleep(0)
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
        
        padding = query.size()[0] - len(request.seqs_data)
        _slot_mapping.extend([-1] * padding)
        for i in range(padding):
            _block_tables.append([0])
        
        block_tables = self._convert_list_to_tensor_padding(_block_tables)
        cpu_attn_metadata.block_tables = block_tables
        cpu_attn_metadata.slot_mapping = torch.tensor(_slot_mapping,
                                                      dtype=torch.int64)

        # 4. Attention computation
        layer = request.layer
        t2 = time.time()
        output = self.attention_unit(query, key, value,
                                     self.cache_enigne.cpu_cache[layer],
                                     cpu_attn_metadata)
        t3 = time.time()
        yield {"result": output.tolist()}
        print(f"att ttl time = {(t3-t1):.6}s, attn time = {(t3-t2):.6}s")

        self._prefetch([self.cache_enigne.cpu_cache[
            (layer+1)%self.model_config.get_num_layers(self.parallel_config)]])
        
    async def get_kv(
        self, 
        request: GetKVRequest
    ) -> dict[int, list[CPU_KVCACHE_DIMENSION]]:
        required_block_hashes = request.cached_hashes

        # content_hash -> kv_cache
        blocks_to_kv: dict[int, List[CPU_KVCACHE_DIMENSION]] = {}

        # Get block_hash related block isd
        block_ids = self.block_manager.get_block_ids_from_hash(required_block_hashes)

        # Get corresponding kv tensors
        for i, block_id in enumerate(block_ids):
            for layer_i in len(self.cache_enigne.cpu_cache):
                layer_tensor = self.cache_enigne.cpu_cache[layer_i][:,block_id,:].copy()
                blocks_to_kv[required_block_hashes[i]].append(layer_tensor.tolist())
        
        return {"kv": blocks_to_kv}