import asyncio
import json
import os
import torch
import aiohttp
from typing import Dict, List, TypeAlias
from vllm.logger import init_logger
from vllm.config import MemPoolConfig
import asyncio
from pydantic import BaseModel

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
    
    async def compute_attention(self, q, k, v) -> None:
        # Check session is on
        if self.session_running is False:
            self.init_session()

        # preprocessing q,k,v

        output = PushdownResponse()
        try:
            global session
            url = self.base_url + "/compute_attention"
            async with session.post() as response:
                if response.status == 200:
                    pass
                else:
                    output.hidden_states = None
                    output.success = False
        except Exception:
            logger.warning(f"pushdown connection timeout, restarting session...")
            self.close_session()
            self.init_session()
            logger.warning(f"restarting session complete")
    
    async def store_kv(
        self, 
        seq_id: int, 
        token_ids: List[int],
        to_transfer_tensor_list: Dict[int, List[KVCAHE_DIMENSION]]
    ) -> None:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                url = self.base_url + "/store_kv"
                payload = {
                    "seq_id": seq_id,
                    "token_ids": token_ids,
                    "tensor_data": to_transfer_tensor_list
                }
                async with session.post(url=url, json=payload) as response:
                    if response.status == 200:
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