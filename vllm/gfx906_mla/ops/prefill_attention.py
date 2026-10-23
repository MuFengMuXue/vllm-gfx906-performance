# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Varlen flash attention for MLA prefill on gfx906.

Adapted from ``vllm/v1/attention/ops/triton_prefill_attention.py``. Three changes:

* separate query and key start offsets, so a shorter query block can attend a longer
  cached context chunk in one call;
* an LSE output, which MLA's chunked-context merge (``merge_attn_states``) needs;
* gfx906 tile sizes: 64 KB of LDS per CU, so upstream's BLOCK_M=64 default does not fit
  at D=256 ((BLOCK_M + 2 * BLOCK_N) * D * 2 bytes of tiles).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2

_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Q_Start,
    Q_Len,
    K_Start,
    K_Len,
    Out,
    Lse,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lh,
    stride_lt,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    q_start = tl.load(Q_Start + cur_batch)
    q_len = tl.load(Q_Len + cur_batch)
    k_start = tl.load(K_Start + cur_batch)
    k_len = tl.load(K_Len + cur_batch)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk

    q = tl.load(
        Q
        + (q_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & mask_d[None, :],
        other=0.0,
    )

    k_ptrs = K + cur_head * stride_kh + offs_d[:, None]
    v_ptrs = V + cur_head * stride_vh + offs_d[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Query positions are offset from the key block by (q_start - k_start); for a
    # new-token prefill the two are equal and this is the usual causal clip.
    q_offset = q_start - k_start
    end_n = k_len
    if IS_CAUSAL:
        end_n = tl.minimum(end_n, q_offset + (start_m + 1) * BLOCK_M)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos_k = start_n + offs_n
        mask = pos_k < k_len
        if IS_CAUSAL:
            mask = mask & (q_offset + offs_m[:, None] >= pos_k[None, :])

        k = tl.load(
            k_ptrs + (k_start + pos_k[None, :]) * stride_kbs,
            mask=(pos_k[None, :] < k_len) & mask_d[:, None],
            other=0.0,
        )
        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + (k_start + pos_k[:, None]) * stride_vbs,
            mask=(pos_k[:, None] < k_len) & mask_d[None, :],
            other=0.0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    out_mask = (offs_m[:, None] < q_len) & mask_d[None, :]
    tl.store(
        Out
        + (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :],
        acc,
        mask=out_mask,
    )
    # exp2-domain m/l -> natural-log LSE, the convention merge_attn_states uses.
    # LSE is [num_heads, num_tokens], as every MLA prefill backend returns it.
    lse = (m_i + tl.math.log2(l_i)) * _LN2
    lse = tl.where(offs_m < q_len, lse, float("-inf"))
    tl.store(
        Lse + cur_head * stride_lh + (q_start + offs_m) * stride_lt,
        lse,
        mask=offs_m < q_len,
    )


def _tile_shape(head_dim: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps) whose tiles fit gfx906's 64 KB of LDS."""
    elem = 2 if dtype.itemsize <= 2 else 4
    # q tile + k tile + v tile, plus slack for the pipeliner.
    budget = (56 * 1024) // elem
    block_n = 32
    block_m = budget // head_dim - 2 * block_n
    block_m = min(32, max(16, block_m // 16 * 16))
    num_warps = 8 if head_dim >= 128 else 4
    return block_m, block_n, num_warps


def varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_start: torch.Tensor,
    q_len: torch.Tensor,
    k_start: torch.Tensor,
    k_len: torch.Tensor,
    max_q_len: int,
    scale: float,
    causal: bool,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Ragged attention over q/k/v laid out as [num_tokens, num_heads, head_dim].

    Args:
        q: [sum(q_len), heads, head_dim] queries, packed by request.
        k: [sum(k_len), heads, head_dim] keys, packed by request (may hold more rows
            than the queries, as for a cached context chunk).
        v: [sum(k_len), heads, head_dim] values.
        q_start: [num_reqs] first row of each request's queries.
        q_len: [num_reqs] query rows per request.
        k_start: [num_reqs] first row of each request's keys.
        k_len: [num_reqs] key rows per request.
        max_q_len: upper bound on q_len, used only to size the grid.
        scale: softmax scale (1/sqrt(head_dim) for these models).
        causal: apply the causal mask, shifted by (q_start - k_start).
        out: optional [sum(q_len), heads, head_dim] destination.
        return_lse: also return the natural-log LSE [heads, sum(q_len)].

    Returns:
        The attention output, or ``(output, lse)`` when ``return_lse`` is set.

    """
    # The platform decides each linear's output dtype: on gfx906 a bf16 model's GEMMs
    # can come back fp16 where no bf16 kernel exists, so q (built by the model) and k/v
    # (built from kv_b_proj inside the impl) may disagree. tl.dot requires one dtype.
    # q is the cheap operand for a context chunk (for new tokens the rows are equal), so
    # it follows k.
    if q.dtype != k.dtype:
        q = q.to(k.dtype)
    assert q.shape[-1] == v.shape[-1], (
        f"q/v head dims differ ({q.shape[-1]} vs {v.shape[-1]}); "
        "pad v or use a backend that supports different head dims"
    )
    head_dim = q.shape[-1]
    num_reqs, num_heads = q_len.shape[0], q.shape[1]
    if out is None:
        out = torch.empty_like(q)
    lse = torch.empty((num_heads, q.shape[0]), dtype=torch.float32, device=q.device)

    block_m, block_n, num_warps = _tile_shape(head_dim, q.dtype)
    grid = (num_reqs, num_heads, triton.cdiv(max_q_len, block_m))
    _fwd_kernel[grid](
        q,
        k,
        v,
        scale * RCP_LN2,
        q_start,
        q_len,
        k_start,
        k_len,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        lse.stride(1),
        BLOCK_M=block_m,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        Lk=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    if return_lse:
        return out, lse
    return out
