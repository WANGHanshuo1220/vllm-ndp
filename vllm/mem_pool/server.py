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

logger = init_logger('vllm.entrypoints.openai.api_server')


async def init_app(args):
    app = FastAPI()
    app.include_router(router)
    app.root_path = args.root_path
    
    pass


async def run_server(args) -> None:
    app = await init_app(args)

    config = uvicorn.Config(app)
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
        logger.info("Gracefully stopping http server")
        return server.shutdown()

if __name__ == "__main__":
    # NOTE(simon):
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    asyncio.run(run_server(args))

