"""Torch fallback for ``gather_and_maybe_dequant_cache`` at non-DeepSeek cache widths.

    RuntimeError: gather_and_maybe_dequant_cache only support the head_dim to
                  320 or 576 for better performance

The compiled kernel hard-rejects any entry size outside DeepSeek's two. Our rows
(``kv_lora_rank + tail``) are outside both, so every chunked-context prefill would die
on them -- which is every request long enough to matter. The impl swaps it in per call.

SEMANTICS, read off ``build_mla_chunked_context_metadata`` rather than guessed:
    chunk_starts[chunk][seq]  -> seq_starts : where this chunk begins IN THE SEQUENCE
    cu_seq_lens[chunk]        -> [0, cumsum(chunk_seq_lens)] : each seq's dst start
    token_to_seq[chunk]       -> repeat_interleave(seq_idx, chunk_seq_lens)
so for dst row j:  s = token_to_seq[j];  tok = seq_starts[s] + (j - cu_seq_lens[s])
and ``tok`` indexes the sequence, which the block table maps into the paged cache.

gfx906 runs bf16 (or fp16) caches only -- FP8 has no hardware conversion here, and
``mla_impl.mla_kv_fp8`` refuses it -- so the plugin's FP8 dequant branches are not
reproduced.
"""

from __future__ import annotations

import torch


def gather_cache_torch(
    *,
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
    mla_rms=None,
) -> None:
    if str(kv_cache_dtype).startswith("fp8"):
        raise NotImplementedError(
            "this port runs bf16/fp16 KV caches only (gfx906 has no FP8 conversion)"
        )
    if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise NotImplementedError(
            f"MLA gather fallback has no path for kv_cache_dtype={kv_cache_dtype}"
        )
    if num_tokens <= 0:
        return

    block_size = src_cache.shape[1]
    row = torch.arange(num_tokens, device=dst.device)
    seq = token_to_seq[:num_tokens].long()
    tok = row - cu_seq_lens.long()[seq]
    if seq_starts is not None:
        tok = tok + seq_starts.long()[seq]
    blocks = block_table[seq, tok // block_size].long()
    dst[:num_tokens].copy_(src_cache[blocks, tok % block_size].reshape(num_tokens, -1))
