from typing import List, Dict, Tuple, Set, TypeAlias
import queue
import torch
import asyncio
import logging
from pydantic import BaseModel

# [2, block_size, num_kv_heads, head_size]
KVCAHE_DIMENSION: TypeAlias = List[List[List[List[float]]]]

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

    def get_new_requests(self) -> List[KVTransferData]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests = []

        while (not self._new_requests.empty()
               and len(new_request) is not self.max_workers):
            new_request = self._new_requests.get()
            new_requests.append(new_request)

        return new_requests
