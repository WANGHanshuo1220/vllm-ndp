import psutil
from typing import List, Dict, Tuple, Optional, AsyncGenerator
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.core.block.block_table import BlockTable
from vllm.worker.cpu_worker import CPUCacheEngine
from vllm.executor.cpu_executor import _verify_and_get_model_config
from vllm.sequence import Sequence, SequenceGroup
from vllm.core.interfaces import AllocStatus
from vllm.utils import init_logger

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, 
                         ParallelConfig, MemPoolConfig, EngineConfig)

import torch

try:
    import rdma_data_struct
except:
    print("No rdma_data_struct found. MemoryPool should be disabled")
    rdma_data_struct = None

logger = init_logger(__name__)

class Shared_mem_resources():

    def __init__(
        self,
        engine_config: EngineConfig,
        connections: int,
    ) -> None:
        self.model_config: ModelConfig = engine_config.model_config
        self.cache_config: CacheConfig = engine_config.cache_config
        self.mem_pool_config: MemPoolConfig = engine_config.mem_pool_config
        self.parallel_config: ParallelConfig = engine_config.parallel_config
        self.device_config: DeviceConfig = engine_config.device_config
        self.connections = connections

        self.dimension = (self.cache_config.block_size *
                          self.model_config.get_num_attention_heads(
                              self.parallel_config) *
                          self.model_config.get_head_size())

        torch.set_default_dtype(torch.bfloat16)
        self.cache_enigne = []
        self.block_manager = []

        # Now we only support tp
        self.tp_size = self.parallel_config.tensor_parallel_size
        for _ in range(self.tp_size):
            self.cache_enigne.append(self._create_cache_engine())
            self.block_manager.append(self._create_block_manager())


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

    
    def _get_reserved_cpu_memory(self):
        """ Reserved memory for RDMA register MR """
        KVCacheMetaSize = rdma_data_struct.KVCacheMetaSize
        KVCacheTensorSize = rdma_data_struct.KVCacheTensorSize
        kv_cache_size = KVCacheMetaSize + KVCacheTensorSize // self.tp_size

        QKVMetaSize = rdma_data_struct.QKVMetaSize
        QKVTensorSize = rdma_data_struct.QKVTensorSize
        qkv_size = QKVMetaSize + QKVTensorSize // self.tp_size

        OutputMetaSize = rdma_data_struct.OutputMetaSize
        OutputTensorSize = rdma_data_struct.OutputTensorSize
        output_size = OutputMetaSize + OutputTensorSize // self.tp_size

        cache_info = rdma_data_struct.CacheInfoSize

        max_size1 = max(kv_cache_size, qkv_size)
        max_size2 = max(output_size, cache_info)

        return self.connections * (max_size1 + max_size2)


    def _get_available_cpu_memory(self):
        memory_info = psutil.virtual_memory()
        return (memory_info.available - 
                self._get_reserved_cpu_memory()) // self.tp_size * \
                self.cache_config.gpu_memory_utilization
    
    """ cache engine functions """
    
    def set_kv_tensor(
        self,
        layer_id: int,
        block_id: int,
        layer_tensor: torch.tensor,
        tp_rank: int,
    ) -> None:
        self.cache_enigne[tp_rank].cpu_cache[layer_id][:,block_id,:] = layer_tensor

    def get_kv_cache_layer(
        self,
        layer_id: int,
        tp_rank: int,
    ) -> torch.tensor:
        return self.cache_enigne[tp_rank].cpu_cache[layer_id]

    
    """ block manager functions """
    
    def get_block_table(
        self,
        seq_id: int,
        tp_rank: int,
    ) -> BlockTable:
        return self.block_manager[tp_rank].block_tables[seq_id]


    def get_blocks(
        self,
        seq: Sequence,
        tp_rank: int,
    ) -> BlockTable:
        return self.block_manager[tp_rank].get_block_table(seq)

    
    def free_seq(
        self,
        seq: Sequence,
        tp_rank: int,
    ) -> None:
        self.block_manager[tp_rank].free(seq)


    def has_seq(
        self,
        seq: Sequence,
        tp_rank: int,
    ) -> bool:
        return self.block_manager[tp_rank].has_seq(seq)

    
    def can_allocate(
        self,
        seq_group: SequenceGroup,
        tp_rank: int,
    ) -> AllocStatus:
        return self.block_manager[tp_rank].can_allocate(seq_group)


    def can_append_slots(
        self,
        seq_group: SequenceGroup,
        num_lookahead_slots: int,
        tp_rank: int,
    ) -> AllocStatus:
        return self.block_manager[tp_rank].can_append_slots(seq_group, 
                                                     num_lookahead_slots)


    def allocate(
        self,
        seq_group: SequenceGroup,
        tp_rank: int,
    ) -> None:
        self.block_manager[tp_rank].allocate(seq_group)


    def append_slots(
        self,
        seq_group: SequenceGroup,
        num_lookahead_slots: int,
        tp_rank: int,
    ) -> None:
        self.block_manager[tp_rank].append_slots(seq_group,
                                        num_lookahead_slots)


    def get_block_reusable(
        self,
        seq: Sequence,
        tp_rank: int,
    ) -> List[bool]:
        return self.block_manager[tp_rank].get_block_reusable(seq)
    
    
    def get_cached_blocks_delta(
        self,
        tp_rank: int,
    ) -> Tuple[List[int], List[int]]:
        return self.block_manager[tp_rank].get_cached_blocks_delta()
    
    