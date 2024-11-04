import asyncio
import json
import os
import torch
import aiohttp
from vllm.logger import init_logger
from vllm.config import MemPoolConfig

logger = init_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

class PushdownResponse:
    hidden_states: torch.tensor
    success: bool

class Attention_pushdown():

    def __init__(self, config: MemPoolConfig) -> None:
        self.session = None
        self.host = config.host
        self.port = config.port
        self.init_session()
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

    def init_session(self):
        self.session = \
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT))

    async def close_session(self):
        await self.session.close()

    async def pushdown_and_retrieve(self, q, k, v) -> PushdownResponse:
        # preprocessing q,k,v

        output = PushdownResponse()
        try:
            async with self.session.post() as response:
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