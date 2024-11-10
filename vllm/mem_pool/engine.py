import psutil
import torch
from typing import List, Dict, TypeAlias
import asyncio

from vllm.attention.layer import Attention
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.worker.cpu_worker import CPUCacheEngine, CPUWorker
from vllm.logger import init_logger
from vllm.sequence import *
from vllm.core.interfaces import AllocStatus

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig, MemPoolConfig, EngineConfig)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from vllm.mem_pool.util import RequestTracker

logger = init_logger('Server execute engine')
KV_TRANSFER_DATA: TypeAlias = Tuple[int, List[int], Dict[int, List[torch.tensor]]]

class Memory_pool_engine():

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 mem_pool_config: MemPoolConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig,) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.mem_pool_config = mem_pool_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        # self.attention_unit = self._create_attention()
        # self.cache_enigne = self._create_cache_engine()
        # self.block_manager = self._create_block_manager()

        self.kv_transfer_running_loop = False
        self._kv_transfer_bg_loop_unshielded: Optional[asyncio.Task] = None
        self.kv_transfer_background_loop: Optional[asyncio.Future] = None
        self.kv_transfer_request_tracker: RequestTracker

        self.attention_running_loop = False
        self.attention_request_tracke: RequestTracker

    def _create_attention(self) -> Attention:
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        head_size = self.model_config.get_head_size()
        scale = 1.0
        return Attention(
            num_heads,
            head_size,
            scale=scale,
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
            enable_caching=self.cache_config.enable_prefix_caching)

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

    def _get_available_cpu_memory(self):
        memory_info = psutil.virtual_memory()
        return memory_info.available * \
            self.cache_config.gpu_memory_utilization

    """ Below are class network interfaces """

    def _can_allocate(self, seq_group) -> bool:
        return self.block_manager.can_allocate(seq_group)

    def _allocate(self) -> None:
        pass

    def _update_radix_tree(self) -> None:
        pass

    @property
    def is_kv_transfer_running(self) -> bool:
        return (self.kv_transfer_background_loop is not None
                and self._kv_transfer_bg_loop_unshielded is not None
                and not self._kv_transfer_bg_loop_unshielded.done())

    async def _running_kv_transfer_running_loop(self) -> None:
        while True:
            await asyncio.sleep(2)
            # await self.kv_transfer_request_tracker.wait_for_new_requests()

    async def _start_kv_transfer_running_loop(self) -> None:

        self.kv_transfer_request_tracker = RequestTracker()
        
        loop = asyncio.get_event_loop()
        self._kv_transfer_bg_loop_unshielded = \
            loop.create_task(self._running_kv_transfer_running_loop())
        self.kv_transfer_background_loop = \
            asyncio.shield(self._kv_transfer_bg_loop_unshielded)
    
    def shutdown_kv_transfer_background_loop(self) -> None:
        if self._kv_transfer_bg_loop_unshielded is not None:
            self._kv_transfer_bg_loop_unshielded.cancel()
            self._kv_transfer_bg_loop_unshielded = None
        self.kv_transfer_background_loop = None

    async def add_kv_transfer_request(
        self, 
        seq_id: int, 
        token_ids: List[int],
        blocks_to_tensor: Dict[int, List[torch.tensor]]
    ) -> None:
        # Start running loop if not
        if self.is_kv_transfer_running is False:
            await self._start_kv_transfer_running_loop()

        # Add this request to waiting queue
        print(f"adding {seq_id} with {token_ids} to request tracker")
        data: KV_TRANSFER_DATA = (seq_id, token_ids, blocks_to_tensor)
        self.kv_transfer_request_tracker.add_request(seq_id, data)

    async def store_kv(self, seq_id: int, token_ids: List[int],
                       blocks_to_tensor: Dict[int, torch.tensor]):
        # Create a sequence group
        sequence = Sequence(
            seq_id=seq_id,
            inputs=LLMInputs(prompt_token_ids=token_ids),
            block_size=self.cache_config.block_size,
        )
        seq_group = SequenceGroup(
            seqs=[sequence],
        )

        # 1. Check if we can allocate blocks for this seq
        can_allocate = self._can_allocate(seq_group)

        # 2. Allocate if we can and free some blocks if neccessary
        if can_allocate == AllocStatus.OK:
            # allocate blocks
            self.block_manager.allocate(seq_group)
        else:
            # free some blocks and allocate
            pass

        # 3. Store tensors and update radix tree

    def compute_attention(self):
        pass