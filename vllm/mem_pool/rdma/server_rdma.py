from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import init_logger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import EngineConfig
import threading

from resources import Shared_mem_resources
from engine import cpu_engine

import rdma_server

logger = init_logger(__name__)

rserver = None

class RDMA_server():

    def __init__(
        self,
        engine_config: EngineConfig,
        num_instances: int,
    ):
        self.num_instances = num_instances
        self.mem_resource = Shared_mem_resources(engine_config)
        self.cpu_engines = []
        for i in range(num_instances):
            self.cpu_engines.append(cpu_engine())

    def start_server(self):
        self.rdma_server = rdma_server.RDMA_Server(self.num_instances)


if __name__=="__main__":
    # Parse args to create configs
    parser = FlexibleArgumentParser(
        description="Memory pool server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Create engine config
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()

    # Create rdma server
    num_instances = 1
    server = RDMA_server(engine_config, num_instances)
