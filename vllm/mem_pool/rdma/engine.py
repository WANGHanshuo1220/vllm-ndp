from vllm.mem_pool.attention import Attention
from vllm.utils import init_logger

logger = init_logger(__name__)

class cpu_engine():
    def __init__(self):
        pass

    def _create_attention(self) -> Attention:
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        model_config = self.model_config.hf_config
        num_kv_heads = getattr(model_config, "num_key_value_heads", 
                               model_config.num_attention_heads), 
        num_kv_heads = num_kv_heads[0]
        head_size = self.model_config.get_head_size()
        hidden_size = self.model_config.get_hidden_size()
        total_num_head = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        head_dim = hidden_size // total_num_head
        scale = head_dim**-0.5
        return Attention(
            num_heads,
            head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=self.cache_config,
            mem_pool_config=self.mem_pool_config
        )