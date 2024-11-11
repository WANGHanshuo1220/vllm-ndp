import asyncio
from contextlib import asynccontextmanager
from typing import Set, List, DefaultDict
import uvicorn
import signal
import time

from fastapi import APIRouter, FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from typing_extensions import assert_never
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from typing import Dict, List, TypeAlias

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.mem_pool.engine import Memory_pool_engine
from vllm.mem_pool.util import StoreKVRequest
import torch

router = APIRouter()
_running_tasks: Set[asyncio.Task] = set()

logger = init_logger(__name__)

engine: Memory_pool_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)

    task = asyncio.create_task(_force_log())
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.remove)

    yield

async def calculate_attention(query: str, context: str):
    # Simulate an attention calculation function
    logger.info(f"Calculating attention for query: {query} with context: {context}")
    # Add the actual attention calculation logic here
    attention_result = {"query": query, "context": context, "attention_score": 0.95}
    return attention_result

@router.get("/check_health")
async def health() -> Response:
    """Health check."""
    print("check health")
    return Response(status_code=200)

@router.post("/store_kv")
def store_kv_cache_endpoint(request: StoreKVRequest):
    global engine
    result = engine.add_kv_transfer_request(request)
    return result

@router.post("/compute_attention")
async def calculate_attention_endpoint(query: str, context: str):
    global engine
    assert engine is not None

    result = await calculate_attention(query, context)
    return result

async def init_app(args):
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path
    return app

async def serve_http(app: FastAPI, **uvicorn_kwargs) -> None:
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        return server.shutdown()

async def run_server(args, **uvicorn_kwargs) -> None:
    
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()

    global engine
    engine = Memory_pool_engine.create(engine_config=engine_config)

    app = await init_app(args)

    shutdown_task = await serve_http(
        app,
        host=args.mp_host,
        port=args.mp_port,
        **uvicorn_kwargs,
    )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Memory pool server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    asyncio.run(run_server(args))

