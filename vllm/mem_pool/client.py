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
        # self.session = None
        self.host = config.host
        self.port = config.port
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        self.session_running = False
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"mp url = {self.base_url}")

    def init_session(self):
        global session
        session = \
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT))
        self.session_running = True

    async def close_session(self):
        global session
        await session.close()

    async def pushdown_and_retrieve(self, q, k, v) -> PushdownResponse:
        # preprocessing q,k,v

        output = PushdownResponse()
        try:
            global session
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
        try:
            global session
            # url = self.base_url + "/health"
            # payload = {
            #     "key": "hello",
            #     "value": "world"
            # }
            url = "http://10.210.22.147:32438/"
            async with session.post(url=url) as response:
                if response.status == 200:
                    response_data = await response.json()
                    print(f"Response Status: {response.status}")
                    print(f"Response Data: {response_data}")
        except Exception:
            pass
            # logger.warning(f"pushdown connection timeout, restarting session...")
            # self.close_session()
            # self.init_session()
            # logger.warning(f"restarting session complete")