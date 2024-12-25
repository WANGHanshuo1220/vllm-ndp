from typing import Dict, List, TypeAlias, Optional, Tuple, Any
import torch
import rdma_client

class rdma_connector():
    def __init__(self):
        pass
    
    def store_kv(
        self,
        engine_id: int,
        seq_ids: List[int],
        seq_lengths: List[int],
        token_ids: List[int],
        free_seq_ids: List[int],
        tensors: List[torch.tensor],
    ) -> None:
        pass

    
    def compute_attention(
        self,
        max_decode_seq_len: int,
        num_decode_tokens: int,
        layer_id: int,
        seq_ids: List[int],
        seq_lengths: List[int],
        token_ids: List[int],
        tensor: List[torch.tensor],
    ) -> None:
        pass