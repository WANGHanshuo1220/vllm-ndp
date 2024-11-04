from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import nullable_str

def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--host",
                        type=nullable_str,
                        default=None,
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")

    return parser