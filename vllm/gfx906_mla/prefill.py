# SPDX-License-Identifier: Apache-2.0
"""MLA prefill backend for gfx906: Triton varlen flash attention over materialised k/v.

gfx906 has no MLA prefill backend in vLLM: the ROCm selector offers AITER flash
attention (gfx90a and newer) and the CUDA flash-attn path, so it raises ``No valid MLA
prefill backend found`` on MI50/MI60/Radeon Pro VII. This backend fills that gap.

The geometry is stock: :mod:`vllm.gfx906_mla.model` narrows q and k to
``(qk_nope_head_dim, qk_rope_head_dim)`` = (192, 64) -- or (198, 58) under partial RoPE
-- so k is ``head_dim`` wide, exactly like v's ``head_dim``. Prefill therefore needs no
MLA-specific kernel at all: two ragged attention calls per chunk, plus the LSE that
``merge_attn_states`` consumes.
"""

import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

from .ops.prefill_attention import varlen_attn
from .ops.prefill_attention import warmup as warmup_prefill

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )

logger = init_logger(__name__)

# (device index, dtype, head dim, num heads): the prefill and merge kernels are
# constexpr-specialised on these, and the backend is cloned per layer.
_WARMED: set[tuple[int, torch.dtype, int, int]] = set()


def _starts_and_lens(cu_seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[n+1] cumulative offsets -> ([n] starts, [n] lengths)."""
    return cu_seq_lens[:-1], cu_seq_lens[1:] - cu_seq_lens[:-1]


def _warmup_merge(
    *, num_heads: int, v_head_dim: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Compile both ``merge_attn_states`` variants before serving."""
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    rows = 8
    partial = torch.zeros((rows, num_heads, v_head_dim), dtype=dtype, device=device)
    lse = torch.full((num_heads, rows), -1.0, dtype=torch.float32, device=device)
    merged = torch.zeros_like(partial)
    for output_lse in (None, torch.empty_like(lse)):
        merge_attn_states(
            output=merged,
            prefix_output=partial,
            prefix_lse=lse,
            suffix_output=partial,
            suffix_lse=lse,
            output_lse=output_lse,
        )


class Gfx906TritonMLAPrefill(MLAPrefillBackend):
    """Ragged Triton flash attention for MLA prefill on gfx906."""

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self._warm_up()

    def _warm_up(self) -> None:
        """Compile the prefill and merge kernels before the first request.

        Neither runs until a prefill has cached context: a prefix-cache hit, or a
        prompt longer than ``max_num_batched_tokens``. Both would otherwise JIT inside
        the first such request -- a latency spike, and a compile landing in the middle
        of graph capture on ROCm.
        """
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float16
        if self.vllm_config is not None:
            dtype = self.vllm_config.model_config.dtype
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        key = (device.index or 0, dtype, head_dim, self.num_heads)
        if key in _WARMED:
            return
        _WARMED.add(key)
        try:
            start = time.perf_counter()
            warmup_prefill(
                head_dim=head_dim,
                num_heads=self.num_heads,
                dtype=dtype,
                device=device,
            )
            _warmup_merge(
                num_heads=self.num_heads,
                v_head_dim=self.v_head_dim,
                dtype=dtype,
                device=device,
            )
            torch.cuda.synchronize()
            logger.info(
                "gfx906 MLA prefill warmup: %s, dtype=%s, head_dim=%d, heads=%d, %.2fs",
                device,
                dtype,
                head_dim,
                self.num_heads,
                time.perf_counter() - start,
            )
        except Exception:
            logger.warning("gfx906 MLA prefill warmup failed", exc_info=True)

    @staticmethod
    def get_name() -> str:
        return "GFX906_TRITON_MLA"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, device_capability) -> bool:
        # gfx906 reports CUDA capability 9.0 through the ROCm shim.
        return device_capability is None or device_capability.major >= 9

    def supports_out(self) -> bool:
        # v is not padded: q and v are both head_dim wide.
        return True

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)
        loc = prefill_metadata.query_start_loc
        self._q_start, self._q_len = _starts_and_lens(loc)

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, "fused FP8 prefill output is not supported"
        return varlen_attn(
            q,
            k,
            v,
            q_start=self._q_start,
            q_len=self._q_len,
            k_start=self._q_start,
            k_len=self._q_len,
            max_q_len=self._prefill_metadata.max_query_len,
            scale=self.scale,
            causal=True,
            out=out,
            return_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_start, q_len = _starts_and_lens(chunk.query_start_loc)
        k_start, k_len = _starts_and_lens(chunk.cu_seq_lens)
        return varlen_attn(
            q,
            k,
            v,
            q_start=q_start,
            q_len=q_len,
            k_start=k_start,
            k_len=k_len,
            max_q_len=chunk.max_query_len,
            scale=self.scale,
            causal=False,
            out=out,
            return_lse=True,
        )
