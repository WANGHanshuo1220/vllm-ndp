import asyncio
import json
import os
import torch
import aiohttp
from vllm.logger import init_logger
from vllm.config import MemPoolConfig
import asyncio

logger = init_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

class PushdownResponse:
    hidden_states: torch.tensor
    success: bool

class Remote_connector():

    def __init__(self, config: MemPoolConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.base_url = f"http://{self.host}:{self.port}"
    
    async def compute_attention(self, q, k, v) -> PushdownResponse:
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
    
    async def store_kv(self, q, k, v) -> PushdownResponse:
        # Check session is on
        if self.session_running is False:
            self.init_session()

        # preprocessing q,k,v

        output = PushdownResponse()
        try:
            global session
            url = self.base_url + "/store_kv"
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
    
    async def dummy_call(self):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                url = self.base_url + "/check_health"
                async with session.get(url=url) as response:
                    if response.status == 200:
                        pass
            except Exception as e:
                print(f"Error: {e}")