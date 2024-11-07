import asyncio
import json
import os
import torch
import aiohttp
from vllm.logger import init_logger
from vllm.config import MemPoolConfig

logger = init_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
session = None

class PushdownResponse:
    hidden_states: torch.tensor
    success: bool

class Attention_pushdown():

    def __init__(self, config: MemPoolConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        self.session_running = False
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"mp url = {self.base_url}")
    
    def __del__(self):
        self.close_session()
        logger.info(f"session is closed gracefully")

    def init_session(self):
        global session
        session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        self.session_running = True

    async def close_session(self):
        global session
        await session.close()
        self.session_running = False

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
    
    async def call(self):
        # Check if session is on
        if self.session_running is False:
            self.init_session()

        try:
            global session
            url = self.base_url + "/check_health"
            async with session.get(url=url) as response:
                if response.status == 200:
                    pass
        except Exception as e:
            print(f"Error: {e}")