from typing import List, Dict, Tuple, Set, TypeAlias
import queue
import torch
import asyncio
from vllm.utils import init_logger
from pydantic import BaseModel
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata

logger = init_logger(__name__)

# [2, block_size, num_kv_heads, head_size]
KVCAHE_DIMENSION: TypeAlias = List[List[List[List[float]]]]

# [num_tokens, token_embedding]
QKV_DIMENSION: TypeAlias = List[List[float]]

class AttentionComputation(BaseModel):
    query: QKV_DIMENSION
    key: QKV_DIMENSION
    value: QKV_DIMENSION
    seq_len_tensor: List[int]
    max_decode_seq_len: int
    num_decode_tokens: int
    seq_lens: List[int]
    seqs_data: Dict[int, List[int]]

class StoreKVRequest(BaseModel):
    seq_id: int
    token_ids: List[int]
    tensor_data: Dict[int, List[KVCAHE_DIMENSION]]

class KVTransferData:

    def __init__(
        self,
        seq_id: int,
        token_ids: List[int],
        blocks_to_tensor: Dict[int, List[torch.tensor]],
    ) -> None:
        self.seq_id = seq_id
        self.token_ids = token_ids
        self.blocks_to_tensor = blocks_to_tensor

class KVRequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self, max_workers: int) -> None:
        self._new_requests = queue.Queue()
        self.all_recieved_requests = []
        self.max_workers = max_workers

    def add_request(self, seq_id: int, content: KVTransferData) -> None:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        self.all_recieved_requests.append(seq_id)
        self._new_requests.put(content)
        logger.debug(f"recieve seq {seq_id}, queue len = {self._new_requests.qsize()}")

    def get_new_requests(self) -> List[KVTransferData]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests = []

        while (not self._new_requests.empty()
               and len(new_requests) is not self.max_workers):
            new_request = self._new_requests.get()
            new_requests.append(new_request)

        return new_requests
