import asyncio
import json
import os
import torch
import aiohttp
from typing import Dict, List, TypeAlias, Optional
from vllm.logger import init_logger
from vllm.config import MemPoolConfig
import asyncio
from pydantic import BaseModel
import time
import requests

from vllm.attention import AttentionMetadata, AttentionType

logger = init_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# [2, block_size, num_kv_heads, head_size]
KVCAHE_DIMENSION: TypeAlias = List[List[List[List[float]]]]

class StoreKVRequest(BaseModel):
    seq_id: int
    token_ids: List[int]
    tensor_data: Dict[int, List[KVCAHE_DIMENSION]]

class Remote_connector():

    def __init__(self, config: MemPoolConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
    
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
        layer: int,
    ) -> torch.Tensor:
        url = self.base_url + "/compute_attention"
        payload = {
            "query": query.tolist(),
            "key": key.tolist(),
            "value": value.tolist(),
            "seq_len_tensor": seq_len_tensor.tolist(),
            "max_decode_seq_len": max_decode_seq_len,
            "num_decode_tokens": num_decode_tokens,
            "seq_lens": seq_lens,
            "seqs_data": seqs_data,
            "layer": layer
        }
        try:
            response = self.session.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return torch.tensor(data["result"])
        except Exception as e:
            print(f"Error: {e}")
    
    def store_kv(
        self, 
        seq_id: int, 
        token_ids: List[int],
        to_transfer_tensor_list: Dict[int, List[KVCAHE_DIMENSION]]
    ) -> None:
        url = self.base_url + "/store_kv"
        payload = {
            "seq_id": seq_id,
            "token_ids": token_ids,
            "tensor_data": to_transfer_tensor_list
        }
        try:
            response = self.session.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                pass
        except Exception as e:
            print(f"Error: {e}")
    
    async def dummy_call(self):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                url = self.base_url + "/check_health"
                async with session.get(url=url) as response:
                    if response.status == 200:
                        pass
            except Exception as e:
                print(f"Error: {e}")