import psutil

from vllm.attention.layer import Attention
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.worker.cpu_worker import CPUCacheEngine

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig, MemPoolConfig, EngineConfig)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

class engine():

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
        # print("==============")
        # print(self.model_config)
        # print("==============")
        # print(self.cache_config)
        # print("==============")
        # print(self.mem_pool_config)
        # print("==============")
        # print(self.parallel_config)
        # print("==============")
        # print(self.device_config)
        # print("==============")

        self.attention_unit = self._create_attention()
        self.block_manager: BlockSpaceManagerV2 = None
        self.cache_enigne: CPUCacheEngine = None
    
    def _create_attention(self) -> Attention:
        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
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
        pass

    def _create_block_manager(self) -> BlockSpaceManagerV2:
        block_size = self.cache_config.block_size
        pass

    @classmethod
    def create(cls, engine_config: EngineConfig) -> "engine":
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
        return memory_info.available

    def store_kv(self):
        pass

    def compute_attention(self):
        pass