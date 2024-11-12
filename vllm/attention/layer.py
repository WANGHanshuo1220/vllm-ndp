"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata, AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, MemPoolConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from vllm.mem_pool.connector import Remote_connector
import asyncio
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    _connector = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        mp_config: Optional[MemPoolConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        mem_pool_config: Optional[MemPoolConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = 1.0
        self._v_scale = 1.0
        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None:
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(num_heads, head_size, num_kv_heads,
                                        sliding_window, dtype, kv_cache_dtype,
                                        block_size, blocksparse_params
                                        is not None)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap)

        self.attn_event_loop = None
        if (mem_pool_config is not None):
            self.attn_event_loop = asyncio.get_event_loop()

        # Create attention pushdown session
        if (mem_pool_config is not None and \
            Attention._connector is None):
            Attention._connector = Remote_connector(mem_pool_config)

    def create_cpu_attn_metadata(
        self,
        attn_metadata: FlashAttentionMetadata
    ) -> TorchSDPAMetadata:
        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_decode_tokens > 0
        cpu_attn_metadata = TorchSDPAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            seq_lens_tensor=attn_metadata.seq_lens_tensor,
            max_decode_seq_len=attn_metadata.max_decode_seq_len,
            block_tables=[],
            num_decode_tokens=attn_metadata.num_decode_tokens,
            slot_mapping=[],
            is_prompt=False,
            seq_lens=attn_metadata.seq_lens
        )
        return cpu_attn_metadata

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
        seq_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if (Attention._connector is not None
            and attn_metadata.decode_metadata is not None
            and seq_ids is not None):

            # TODO:Here we need to invoke remote memory pool
            # to compute attention. 
            # [(q,k,v), (engine_id, req_id)] need to be transfered,
            # [kv_cache, attn_metadata] is managed by remote pool

            assert self.attn_event_loop is not None

            # 1. Create a cpu-based attn_metadata for decode
            cpu_attn_metadata = self.create_cpu_attn_metadata(attn_metadata)

            # 2. Transfer q,k,v and cpu_attn_metadata and get result
            # task = self.attn_event_loop.create_task(
            #     Attention._connector.dummy_call()
            # )
            # self.attn_event_loop.run_until_complete(task)

        # q, k, v are all [num_tokens, embedding_size]
        return self.impl.forward(query,
                                 key,
                                 value,
                                 kv_cache,
                                 attn_metadata,
                                 self._k_scale,
                                 self._v_scale,
                                 attn_type=attn_type)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s
