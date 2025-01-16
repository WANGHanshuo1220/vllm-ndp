from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import init_logger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import EngineConfig
import threading

from resources import Shared_mem_resources
from engine import cpu_engine
from uuid import uuid4
import os
import time

try:
    import rdma_server
except:
    print("No rdma_server found. MemoryPool should be disabled")
    rdma_server = None

try:
    import rdma_data_struct
except:
    print("No rdma_data_struct found. MemoryPool should be disabled")
    rdma_data_struct = None

logger = init_logger(__name__)

MEM_POOL_ID = None

def log_to_file(file_name, log_message):
    log_message += f" ({time.time()})"
    with open(file_name, "a", buffering=1) as log_file:
        log_file.write(log_message + "\n")

def client_loop(
    server: rdma_server.RDMA_Server, 
    engine_config: EngineConfig,
    shared_resources: Shared_mem_resources,
    connection_id: int
) -> None:
    # Setup connections
    server.setup_client_resources(connection_id)
    server.accept_client_connection(connection_id)

    server.wait_completion_event(1, connection_id)

    # Engine connection info
    engine_id = server.get_engine_id(connection_id)
    tp_rank = server.get_tp_rank(connection_id)

    # Create a CPU engine
    engine = cpu_engine(engine_config, shared_resources, tp_rank)
    server.prepare_recv_data_wr(connection_id)
    server.recv_data_from_client(connection_id)

    global MEM_POOL_ID
    server.send_server_metadata_to_client(connection_id, MEM_POOL_ID)
    server.wait_completion_event(1, connection_id)

    server.prepare_send_data_wr(connection_id)
    recv_kv_cache_handler = server.get_recv_kv_cache_handler(connection_id)
    send_cache_info_handler = server.get_send_cache_info_handler(connection_id)
    recv_qkv_handler = server.get_recv_qkv_handler(connection_id)
    send_output_handler = server.get_send_output_handler(connection_id)

    first = True
    while (not server.check_client_disconnected(connection_id)):

        if first:
            server.wait_completion_event(1, connection_id)
            server.update_client_status(connection_id)
            if (server.check_client_disconnected(connection_id)):
                break
            first = False

        # Processing data
        if (server.is_prefill_kv_cache(connection_id)):
            engine.save_kv_cache(recv_kv_cache_handler, send_cache_info_handler)
            # recv_handler.pretty_print()
            # send_handler.pretty_print()
        else:
            engine.compute_attention(recv_qkv_handler, send_output_handler)
            # recv_handler.pretty_print()
            # send_handler.pretty_print()

        server.send_data_to_client(connection_id)
        server.recv_data_from_client(connection_id)
        server.wait_completion_event(2, connection_id)

        server.update_client_status(connection_id)

    server.disconnect_and_cleanup(connection_id)

def set_mempool_id() -> None:
    global MEM_POOL_ID
    MEM_POOL_ID = str(uuid4().hex)
    MEM_POOL_ID = hash(MEM_POOL_ID) & 0xFFFFFFFF

if __name__=="__main__":
    # Parse args to create configs
    parser = FlexibleArgumentParser(
        description="Memory pool server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Create engine config
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()

    # Get connections in this cluster
    num_engine = 1
    tp_size = engine_config.parallel_config.tensor_parallel_size
    pp_size = engine_config.parallel_config.pipeline_parallel_size
    num_connections = tp_size * pp_size * num_engine

    # Set mempool id
    set_mempool_id()

    # Create shared resources (cache_engine and block_manager)
    shared_resources = Shared_mem_resources(engine_config,
                                            num_connections)

    # Create rdma server
    server = rdma_server.RDMA_Server(tp_size, pp_size, num_engine)
    server.start_rdma_server(3389)

    threads = []
    for i in range(num_connections):
        thread = threading.Thread(target=client_loop, 
                                  args=(server, engine_config, 
                                        shared_resources, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    server.close_server()    
