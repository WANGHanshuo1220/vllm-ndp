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

def client_loop(
    server: rdma_server.RDMA_Server, 
    engine_config: EngineConfig,
    shared_resources: Shared_mem_resources,
    client_id: int
) -> None:
    # Create a CPU engine
    engine = cpu_engine(engine_config, shared_resources)
    
    server.setup_client_resources(client_id)
    server.accept_client_connection(client_id)

    server.wait_completion_event(1, client_id)

    server.prepare_recv_data_wr(client_id)
    server.recv_data_from_client(client_id)

    server.send_server_metadata_to_client(client_id)
    server.wait_completion_event(1, client_id)

    server.prepare_send_data_wr(client_id)
    first = True
    while (not server.check_client_disconnected(client_id)):

        if first:
            server.wait_completion_event(1, client_id)
            server.update_client_status(client_id)
            if (server.check_client_disconnected(client_id)):
                break
            first = False

        # Processing data
        if (server.is_prefill_kv_cache(client_id)):
            print("recieve a prefill")
            # This is a prefill save kv cache request
            recv_handler = server.get_recv_kv_cache_handler(client_id)
            send_handler = server.get_send_cache_info_handler(client_id)
            engine.save_kv_cache(recv_handler, send_handler)
            # recv_handler.pretty_print()
            # send_handler.pretty_print()
        else:
            print("recieve a decode")
            # This is a decode attention computation request
            recv_handler = server.get_recv_qkv_handler(client_id)
            send_handler = server.get_send_output_handler(client_id)
            engine.compute_attention(recv_handler, send_handler)
            # recv_handler.pretty_print()
            # send_handler.pretty_print()

        server.send_data_to_client(client_id)
        server.recv_data_from_client(client_id)
        server.wait_completion_event(2, client_id)

        server.update_client_status(client_id)

    server.disconnect_and_cleanup(client_id)

if __name__=="__main__":
    # Parse args to create configs
    parser = FlexibleArgumentParser(
        description="Memory pool server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Create engine config
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()

    # Create shared resources (cache_engine and block_manager)
    shared_resources = Shared_mem_resources(engine_config)

    # Create rdma server
    num_instances = 1
    server = rdma_server.RDMA_Server(num_instances)
    server.start_rdma_server(3389)

    threads = []
    for i in range(num_instances):
        thread = threading.Thread(target=client_loop, 
                                  args=(server, engine_config, 
                                        shared_resources, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    server.close_server()    
