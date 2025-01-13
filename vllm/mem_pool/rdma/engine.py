from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, 
                         ParallelConfig, MemPoolConfig, EngineConfig)
from vllm.sequence import Sequence, SequenceGroup
from vllm.mem_pool.attention import Attention
from vllm.core.interfaces import AllocStatus
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata
from vllm.utils import init_logger

import time
import torch
from typing import List

from resources import Shared_mem_resources
import rdma_data_struct

logger = init_logger(__name__)

def log_to_file(file_name, log_message):
    log_message += f" ({time.time()})"
    with open(file_name, "a") as log_file:
        log_file.write(log_message + "\n")

class cpu_engine():
    def __init__(
        self, 
        engine_config: EngineConfig,
        shared_resources: Shared_mem_resources,
        tp_rank: int,
    ) -> None:
        self.model_config: ModelConfig = engine_config.model_config
        self.mem_pool_config: MemPoolConfig = engine_config.mem_pool_config
        self.parallel_config: ParallelConfig = engine_config.parallel_config
        self.cache_config: CacheConfig = engine_config.cache_config

        self.tp_rank = tp_rank
        self.tp_size = self.parallel_config.tensor_parallel_size
        self.file_name = f"output_{tp_rank}.log"

        self.cpu_kv_dimension = \
            (self.cache_config.block_size *
             self.model_config.get_num_attention_heads(
                 self.parallel_config) *
             self.model_config.get_head_size())
        print(f"{self.cpu_kv_dimension=}")

        torch.set_default_dtype(torch.bfloat16)
        self.shared_resources = shared_resources
        self.attention_unit = self._create_attention()


    def _create_attention(self) -> Attention:
        self.num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )

        self.total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)

        self.head_dim = self.model_config.get_head_size()
        
        scale = self.head_dim**-0.5
        return Attention(
            self.num_heads,
            self.head_dim,
            scale=scale,
            num_kv_heads=self.num_kv_heads,
            cache_config=self.cache_config,
            mem_pool_config=self.mem_pool_config
        )


    def _store_kv_tensor(
        self,
        seq_id: int,
        allocated_blocks: List[int],
        blocks_reusable: List[bool],
        blocks_tensor: List[List[torch.tensor]], # blocks -> layers
    ) -> None:
        for i, layer_tensors in enumerate(blocks_tensor):
            block_id = allocated_blocks[i]
            if not blocks_reusable[i]:
                for j in range(len(layer_tensors)):
                    tensor = layer_tensors[j].view(2, self.cpu_kv_dimension)
                    self.shared_resources.set_kv_tensor(j, block_id, tensor, self.tp_rank)


    def save_kv_cache(
        self,
        recv_handler: rdma_data_struct.ClientSendKVCache,
        send_handler: rdma_data_struct.ClientRecvCacheInfo,
    ):
        log_to_file(self.file_name, f"Begin save kv cache")
        seq_ids = recv_handler.get_seq_ids()
        token_ids = recv_handler.get_token_ids()
        tensor_list = recv_handler.get_tensor()
        to_free_seq_ids = recv_handler.get_to_free_seq_ids()
        seq_lengths = recv_handler.get_seq_lengths()
        log_to_file(self.file_name, f"{seq_ids=}, {token_ids=}, {seq_lengths=}")

        assert(len(seq_ids) == len(seq_lengths))
        assert(len(seq_ids) == len(tensor_list), 
            f"{len(seq_ids)=} vs. {len(tensor_list)=}")
        assert(len(token_ids) == sum(seq_lengths))

        # First free blocks of finished seqs
        for to_free_seq_id in to_free_seq_ids:
            block_table = self.shared_resources.get_block_table(to_free_seq_id, self.tp_rank)
            to_free_token_ids = block_table._get_all_token_ids()
            to_free_sequence = Sequence(
                seq_id=to_free_seq_id,
                inputs={"prompt_token_ids": to_free_token_ids},
                block_size=self.cache_config.block_size,
            )
            self.shared_resources.free_seq(to_free_sequence, self.tp_rank)

        seqs_token_ids = []
        start = 0
        for length in seq_lengths:
            end = start + length
            seqs_token_ids.append(token_ids[start:end])
            start = end

        assert(len(seqs_token_ids) == len(seq_ids))

        log_to_file(self.file_name, f"in iter")
        for i, seq_id in enumerate(seq_ids):
            log_to_file(self.file_name, f"  {i=}: {seq_id=}")

            seq_token_ids = seqs_token_ids[i]
            seq_kv_blocks = tensor_list[i]

            # Create a sequence group
            sequence = Sequence(
                seq_id=seq_id,
                inputs={"prompt_token_ids": seq_token_ids},
                block_size=self.cache_config.block_size,
            )
            seq_group = SequenceGroup(
                request_id=seq_id,
                seqs=[sequence],
                arrival_time=time.time()
            )

            # 1. Check if we can allocate blocks for this seq
            can_allocate = self.shared_resources.can_allocate(seq_group, self.tp_rank)

            # 2. Allocate if we can and free some blocks if neccessary
            if can_allocate == AllocStatus.OK:
                # allocate blocks
                self.shared_resources.allocate(seq_group, self.tp_rank)
            elif can_allocate == AllocStatus.LATER:
                assert False, "Store KV later is not implemented yet"
            else:
                assert False, "Abort exception is not implemented yet"

            # 3. Store tensors to cpu cache
            allocated_blocks = self.shared_resources.get_blocks(sequence, self.tp_rank)
            blocks_reusable = self.shared_resources.get_block_reusable(sequence, self.tp_rank)
            assert len(allocated_blocks) == len(blocks_reusable)
            assert len(allocated_blocks) == len(seq_kv_blocks)

            self._store_kv_tensor(
                seq_id, allocated_blocks, blocks_reusable, seq_kv_blocks
            )

        add_delta, pop_delta = self.shared_resources.get_cached_blocks_delta(self.tp_rank)
        send_handler.set_all(add_delta, pop_delta)

            
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


    def compute_attention(
        self,
        recv_handler: rdma_data_struct.ClientSendQKV,
        send_handler: rdma_data_struct.ClientRecvOutput,
    ) -> None:
        qkv = recv_handler.get_tensor()
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        seq_lens = recv_handler.get_seq_lengths()
        seq_len_tensor = torch.tensor(seq_lens, dtype=torch.int32)
        max_decode_seq_len = recv_handler.get_max_decode_seq_len()
        num_decode_tokens = recv_handler.get_num_decode_tokens()

        cpu_attn_metadata = self._create_cpu_attn_metadata(
            seq_lens_tensor=seq_len_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens
        )

        seq_ids = recv_handler.get_seq_ids()
        all_token_ids = recv_handler.get_token_ids()
        seq_lengths = recv_handler.get_seq_lengths()
        assert(len(seq_ids) == len(seq_lengths))

        seqs_token_ids = []
        start = 0
        for length in seq_lengths:
            end = start + length
            seqs_token_ids.append(all_token_ids[start:end])
            start = end
        assert(len(seq_ids) == len(seqs_token_ids))

        _block_tables = []
        _slot_mapping = []
        for i in range(len(seq_ids)):
            seq_id = seq_ids[i]
            token_ids = seqs_token_ids[i]

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
            while not self.shared_resources.has_seq(sequence, self.tp_rank):
                assert False, "seqs does not exist"
            else:
                assert self.shared_resources.can_append_slots(
                    seq_group, num_lookahead_slots=0, tp_rank=self.tp_rank)
            
                self.shared_resources.append_slots(
                    sequence, num_lookahead_slots=0, tp_rank=self.tp_rank)

            # 3. Get block info and slot_mapping to attn_matadata
            block_ids = self.shared_resources.get_blocks(sequence, self.tp_rank)
            _block_tables.append(block_ids)

            offset = seq_lens[i] 
            if offset > self.cache_config.block_size:
                offset = offset % self.cache_config.block_size

            slot_mapping = (block_ids[-1] * self.cache_config.block_size
                            + offset - 1)
            _slot_mapping.append(slot_mapping)

        padding = query.size()[0] - len(seq_ids)
        _slot_mapping.extend([-1] * padding)
        for i in range(padding):
            _block_tables.append([0])
        
        block_tables = self._convert_list_to_tensor_padding(_block_tables)
        cpu_attn_metadata.block_tables = block_tables
        cpu_attn_metadata.slot_mapping = torch.tensor(_slot_mapping,
                                                      dtype=torch.int64)

        # 4. Attention computation
        layer = recv_handler.get_layer_id()
        output = self.attention_unit(
            query, key, value,
            self.shared_resources.get_kv_cache_layer(layer, self.tp_rank),
            cpu_attn_metadata)
        send_handler.set_all(query.shape[0], output)