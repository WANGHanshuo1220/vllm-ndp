from typing import Dict, List, TypeAlias, Optional, Tuple, Any
from vllm.config import MemPoolConfig
import torch
import rdma_client
import rdma_data_struct
import time

BLOCK_SIZE = rdma_data_struct.BLOCK_SIZE
MAX_BATCH_SIZE = rdma_data_struct.MAX_BATCH_SIZE
MAX_SEQ_LENGTH = rdma_data_struct.MAX_SEQ_LENGTH
HIDDEN_SIZE = rdma_data_struct.HIDDEN_SIZE
NUM_HEADS = rdma_data_struct.NUM_HEADS
NUM_LAYER = rdma_data_struct.NUM_LAYER

def log_to_file(file_name, log_message):
    log_message += f" ({time.time()})"
    with open(file_name, "a") as log_file:
        log_file.write(log_message + "\n")

class RemoteConnector():
    def __init__(
        self, 
        config: Optional[MemPoolConfig] = None,
        engine_id: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        self.engine_id = engine_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        file_name = f"connector_output_{tp_rank}.log"
        with open(file_name, "w") as log_file:
            log_file.write("")

        self.client = rdma_client.RDMA_Client(self.tp_size)
        log_to_file(file_name, f"{tp_rank=} create client success")
        self.client.client_prepare_connection(
            int(config.port), config.host)
        log_to_file(file_name, f"{tp_rank=} prepare connection success")
        self.client.client_pre_post_recv()
        self.client.client_connect_to_server()
        log_to_file(file_name, f"{tp_rank=} connect to server success")
        self.client.client_xchange_metadata_with_server(
            self.engine_id, self.tp_rank)
        log_to_file(file_name, f"{tp_rank=} xchange metadata success")

        self.prefill_handler = self.client.get_send_kv_cache_handler()
        self.decode_handler = self.client.get_send_qkv_handler()
        self.output_handler = self.client.get_recv_output_handler()
        self.cache_info_handler = self.client.get_recv_cache_info_handler()
    
    def __del__(self):
        self.client.client_disconnect_and_clean()
    
    def store_kv(
        self,
        seq_ids: List[int],
        seq_lengths: List[int],
        token_ids: List[int],
        free_seq_ids: List[int],
        tensors: List[List[List[torch.tensor]]],
    ) -> Tuple[List[int], List[int]]:
        self.client.set_prefill()

        assert(len(seq_ids) == len(seq_lengths))
        assert(len(token_ids) == sum(seq_lengths))

        assert(len(tensors) == len(seq_ids))
        for i, seq_length in enumerate(seq_lengths):
            num_blocks = (seq_length + BLOCK_SIZE - 1) // BLOCK_SIZE
            assert(len(tensors[i]) == num_blocks)
            for block_layers in tensors[i]:
                assert(len(block_layers) == NUM_LAYER)
                for t in block_layers:
                    assert t.shape == (2, BLOCK_SIZE, NUM_HEADS//self.tp_size, HIDDEN_SIZE//NUM_HEADS)

        self.prefill_handler.set_all(
            self.tp_size,
            self.engine_id,
            self.tp_rank,
            seq_ids,
            seq_lengths,
            token_ids,
            free_seq_ids,
            tensors
        )
        self.client.client_remote_data_processing()
        add_delta = self.cache_info_handler.get_add_delta()
        pop_delta = self.cache_info_handler.get_pop_delta()
        return (add_delta, pop_delta)
    
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
            self.tp_size,
            self.tp_rank,
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

def run_prefill(client):
    engine_id = 1
    seq_ids = [0]
    seq_lengths = [42]
    token_ids = [i+100 for i in range(seq_lengths[0])]
    free_seq_ids = []

    seq_num_blocks = []
    for seq_length in seq_lengths:
        seq_num_blocks.append((seq_length + BLOCK_SIZE - 1) // BLOCK_SIZE)

    input_tensor = []
    for i in range(len(seq_ids)):
        num_blocks = seq_num_blocks[i]
        seq_blocks = []
        for block_idx in range(num_blocks):
            block_layers = []
            for j in range(NUM_LAYER):
                layer_tensor = torch.rand(
                    [2, BLOCK_SIZE, NUM_HEADS, HIDDEN_SIZE//NUM_HEADS],
                    dtype=torch.float16
                )
                block_layers.append(layer_tensor)
            seq_blocks.append(block_layers)
        input_tensor.append(seq_blocks)
    
    client.store_kv(
        engine_id,
        seq_ids,
        seq_lengths,
        token_ids,
        free_seq_ids,
        input_tensor
    )
    
def run_decode(client):
    max_decode_seq_len = 9
    num_decode_tokens = 3
    layer_id = 0
    seq_ids = [0, 1, 2]
    seq_lengths = [4, 7, 9]
    token_ids = [1, 2, 3, 4,
                 1, 2, 3, 4, 5, 6, 7,
                 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    seq_data = {}
    start = 0
    for i in range(len(seq_ids)):
        end = start + seq_lengths[i]
        seq_data[i] = token_ids[start:end]
        start = end

    input_tensor = []
    for i in range(3):
        tensor = torch.rand(
            [len(seq_ids), HIDDEN_SIZE],
            dtype=torch.float16
        )
        input_tensor.append(tensor)

    client.compute_attention(
        input_tensor[0],
        input_tensor[1],
        input_tensor[2],
        torch.tensor(seq_lengths),
        max_decode_seq_len,
        num_decode_tokens,
        seq_lengths,
        seq_data,
        layer_id,
    )

if __name__=="__main__":
    config = MemPoolConfig.may_be_create(True, "172.16.253.36", "3389")
    client = RemoteConnector(config) 

    # prefill
    run_prefill(client)
    
    # decode
    # run_decode(client)