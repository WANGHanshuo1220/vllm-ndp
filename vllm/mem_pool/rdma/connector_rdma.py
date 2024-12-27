from typing import Dict, List, TypeAlias, Optional, Tuple, Any
from vllm.config import MemPoolConfig
import torch
import rdma_client

class Remote_connector():
    def __init__(self, config: MemPoolConfig):
        self.client = rdma_client.RDMA_Client()
        self.client.client_prepare_connection(
            config.port, config.host)
        self.client.client_pre_post_recv()
        self.client.client_connect_to_server()
        self.client.client_xchange_metadata_with_server()

        self.prefill_handler = self.client.get_send_kv_cache_handler()
        self.decode_handler = self.client.get_send_qkv_handler()
        self.output_handler = self.client.get_recv_output_handler()
    
    def __del__(self):
        self.client.client_disconnect_and_clean()
    
    def store_kv(
        self,
        engine_id: int,
        seq_ids: List[int],
        seq_lengths: List[int],
        token_ids: List[int],
        free_seq_ids: List[int],
        tensors: List[List[List[torch.tensor]]],
    ) -> None:
        self.client.set_prefill()
        self.prefill_handler.set_all(
            engine_id,
            seq_ids,
            seq_lengths,
            token_ids,
            free_seq_ids,
            tensors
        )
        self.client.client_remote_data_processing()
    
    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        max_decode_seq_len: int,
        num_decode_tokens: int,
        seq_lens: List[int],
        seqs_data: Dict[int, List[int]],
        layer_id: int,
    ) -> torch.tensor:
        self.client.set_decode()
        
        seq_ids = list(seqs_data.keys())
        token_ids = [item 
                     for sublist in seqs_data.values() 
                     for item in sublist] 
        tensor = [query, key, value]

        self.decode_handler.set_all(
            max_decode_seq_len,
            num_decode_tokens,
            layer_id,
            seq_ids,
            seq_lens,
            token_ids,
            tensor
        )

        self.client.client_remote_data_processing()
        return self.output_handler.get_tensor()