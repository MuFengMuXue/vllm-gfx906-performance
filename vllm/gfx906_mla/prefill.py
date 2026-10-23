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

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

from .ops.prefill_attention import varlen_attn

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


def _starts_and_lens(cu_seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[n+1] cumulative offsets -> ([n] starts, [n] lengths)."""
    return cu_seq_lens[:-1], cu_seq_lens[1:] - cu_seq_lens[:-1]


class Gfx906TritonMLAPrefill(MLAPrefillBackend):
    """Ragged Triton flash attention for MLA prefill on gfx906."""

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
