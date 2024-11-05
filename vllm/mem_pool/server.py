import asyncio
import importlib
import inspect
import multiprocessing
import os
import re
import tempfile
from argparse import Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncIterator, Optional, Set
import uvicorn
import signal

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount
from typing_extensions import assert_never
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from vllm.mem_pool.util import make_arg_parser

router = APIRouter()
_running_tasks: Set[asyncio.Task] = set()

logger = init_logger('vllm.entrypoints.openai.api_server')

@asynccontextmanager
async def lifespan(app: FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)

    task = asyncio.create_task(_force_log())
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.remove)

    yield

async def store_kv_cache(key: str, value: str):
    # Here we simulate storing a key-value pair, such as in a cache or database
    logger.info(f"Storing KV pair: {key} -> {value}")
    # Add the actual storage logic here
    return {"status": "success", "key": key, "value": value}


async def calculate_attention(query: str, context: str):
    # Simulate an attention calculation function
    logger.info(f"Calculating attention for query: {query} with context: {context}")
    # Add the actual attention calculation logic here
    attention_result = {"query": query, "context": context, "attention_score": 0.95}
    return attention_result

@router.get("/")
async def empty() -> Response:
    """Health check."""
    print("recieve")
    return Response(status_code=200)

@router.get("/health")
async def health() -> Response:
    """Health check."""
    print("check health")
    return Response(status_code=200)

@router.post("/store_kv_cache")
async def store_kv_cache_endpoint(key: str, value: str):
    result = await store_kv_cache(key, value)
    return result


@router.post("/calculate_attention")
async def calculate_attention_endpoint(query: str, context: str):
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
    app = await init_app(args)

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
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

