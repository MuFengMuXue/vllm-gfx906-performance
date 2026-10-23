# SPDX-License-Identifier: Apache-2.0
"""Numerics gates for the vendored MLA kernels on gfx906.

Two kernels carry the retrofit end to end:

* ``ops/mla_decode_attention.py`` -- absorbed decode, with Qwen3.5's per-(key, head) rms
  divisor folded into the logits (DeepSeek's latent normalisation makes it unnecessary);
* ``ops/prefill_attention.py`` -- the prefill backend vLLM does not ship here.

Both are checked against an explicit torch reference written in the kernel's own order:
``qk = (q_latent . k_latent + q_rope . k_rope) / rms``, then ``* sm_scale``,
then softmax.

Tolerances are bf16-accumulation tolerances (the softmax denominator is fp32 but the
products are not), not fudge factors: a tiling change that moves the result by more than
these is the exact class of bug this file exists to catch.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.gfx906_mla.ops.mla_decode_attention import decode_attention_fwd
from vllm.gfx906_mla.ops.prefill_attention import varlen_attn
from vllm.gfx906_mla.prefill import Gfx906TritonMLAPrefill

PAGE = 400  # a real MLA page size: the kernel requires a tile to fit inside one page

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


# --------------------------------------------------------------------------------------
# decode: latent + packed rope, divided by the per-(token, head) rms tail
# --------------------------------------------------------------------------------------


def _decode_reference(
    q, kv, block_table, seq_lens, rank, packed_rope, num_heads, scale
):
    batch, heads, _ = q.shape
    out = torch.zeros(batch, heads, rank, dtype=torch.float32, device=q.device)
    qk_dim = rank + packed_rope
    for b in range(batch):
        length = int(seq_lens[b])
        rows = [kv[int(block_table[b, t // PAGE]), t % PAGE, 0] for t in range(length)]
        keys = torch.stack(rows).float()
        k_qk = keys[:, :qk_dim]
        rms = keys[:, qk_dim : qk_dim + num_heads]  # [L, H]
        for h in range(heads):
            scores = (q[b, h].float() @ k_qk.T) / rms[:, h].clamp(min=1e-6)
            probs = torch.softmax(scores * scale, dim=-1)
            out[b, h] = probs @ keys[:, :rank]
    return out


def _run_decode(q, kv, block_table, seq_lens, rank, packed_rope, splits):
    batch, heads, _ = q.shape
    out = torch.zeros(batch, heads, rank, dtype=torch.bfloat16, device=q.device)
    lse = torch.zeros(batch, heads, dtype=torch.float32, device=q.device)
    logits = torch.zeros(
        batch, heads, splits, rank + 1, dtype=torch.float32, device=q.device
    )
    decode_attention_fwd(
        q,
        kv,
        kv[..., :rank],
        out,
        lse,
        block_table,
        seq_lens,
        logits,
        splits,
        1.0 / math.sqrt(256),
        PAGE,
        is_mla=True,
        rms_offset=rank + packed_rope,
    )
    torch.cuda.synchronize()
    return out.float()


@pytest.mark.parametrize(
    "rank,packed_rope,heads",
    [
        (256, 256, 24),  # narrow layer, full rope, TP=1
        (512, 128, 24),  # middle layer
        (896, 116, 12),  # widest layer of a GLA-g2 checkpoint at TP=2 (partial rope)
    ],
)
def test_decode_kernel_matches_reference(rank, packed_rope, heads):
    ctx, batch, splits = 900, 2, 8
    dev = torch.device("cuda:0")
    torch.manual_seed(rank + heads)
    entry = rank + packed_rope + heads
    pages_per_seq = (ctx + PAGE - 1) // PAGE
    kv = (
        torch.randn(
            batch * pages_per_seq + 4, PAGE, 1, entry, dtype=torch.bfloat16, device=dev
        )
        * 0.05
    )
    kv[..., rank + packed_rope :] = kv[..., rank + packed_rope :].abs() + 0.3  # rms > 0
    block_table = torch.arange(
        batch * pages_per_seq, device=dev, dtype=torch.int32
    ).reshape(batch, pages_per_seq)
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    q = (
        torch.randn(batch, heads, rank + packed_rope, dtype=torch.bfloat16, device=dev)
        * 0.05
    )

    ref = _decode_reference(
        q, kv, block_table, seq_lens, rank, packed_rope, heads, 1.0 / math.sqrt(256)
    )
    err = _rel(
        _run_decode(q, kv, block_table, seq_lens, rank, packed_rope, splits), ref
    )
    assert err < 5e-2, f"rank {rank}: rel err {err:.3e}"


def test_decode_kernel_tiles_agree(monkeypatch):
    """BLOCK_H / BLOCK_N / num_warps are pure tiling: they must not move the result."""
    rank, packed_rope, heads = 512, 128, 12
    ctx, batch = 600, 1
    dev = torch.device("cuda:0")
    torch.manual_seed(7)
    entry = rank + packed_rope + heads
    pages = (ctx + PAGE - 1) // PAGE
    kv = torch.randn(2 * pages, PAGE, 1, entry, dtype=torch.bfloat16, device=dev) * 0.05
    kv[..., rank + packed_rope :] = kv[..., rank + packed_rope :].abs() + 0.3
    block_table = torch.arange(2 * pages, device=dev, dtype=torch.int32).reshape(
        batch, -1
    )
    seq_lens = torch.full((batch,), ctx - 3, dtype=torch.int32, device=dev)
    q = (
        torch.randn(batch, heads, rank + packed_rope, dtype=torch.bfloat16, device=dev)
        * 0.05
    )

    default = _run_decode(q, kv, block_table, seq_lens, rank, packed_rope, 8)
    for env in ({"QWEN_MLA_BLOCK_H": "8"}, {"QWEN_MLA_BLOCK_N": "16"}):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        alt = _run_decode(q, kv, block_table, seq_lens, rank, packed_rope, 8)
        for key in env:
            monkeypatch.delenv(key)
        err = _rel(alt, default)
        assert err < 2e-2, f"{env}: tiling changed the result by {err:.3e}"


# --------------------------------------------------------------------------------------
# prefill: ragged flash attention with an LSE, over materialised k/v
# --------------------------------------------------------------------------------------


def _prefill_reference(q, k, v, q_start, q_len, k_start, k_len, scale, causal):
    outs, lses = [], []
    for b in range(q_len.numel()):
        nq, nk = int(q_len[b]), int(k_len[b])
        qs, ks = int(q_start[b]), int(k_start[b])
        qb = q[qs : qs + nq].float()
        kb = k[k_start[b] : k_start[b] + nk].float()
        vb = v[k_start[b] : k_start[b] + nk].float()
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        if causal:
            offset = qs - ks
            visible = (
                torch.arange(nq, device=q.device)[:, None] + offset
                >= torch.arange(nk, device=q.device)[None, :]
            )
            scores = scores.masked_fill(~visible, float("-inf"))
        lses.append(torch.logsumexp(scores, -1).T)
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vb))
    return torch.cat(outs), torch.cat(lses)


@pytest.mark.parametrize(
    "q_lens,k_lens,causal",
    [
        ([64, 37, 128], [64, 37, 128], True),  # new tokens: q and k are the same rows
        ([64, 37], [200, 160], False),  # cached context chunk: q is shorter than k
        ([33], [33], False),
    ],
)
def test_prefill_kernel_matches_reference(q_lens, k_lens, causal):
    heads, dim = 12, 256
    dev = torch.device("cuda:0")
    torch.manual_seed(sum(q_lens) + heads)
    scale = 1.0 / math.sqrt(dim)
    q = torch.randn(sum(q_lens), heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    k = torch.randn(sum(k_lens), heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(sum(k_lens), heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    q_start = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)[:-1]), device=dev, dtype=torch.int32
    )
    k_start = torch.tensor(
        [0] + list(torch.tensor(k_lens).cumsum(0)[:-1]), device=dev, dtype=torch.int32
    )
    out, lse = varlen_attn(
        q,
        k,
        v,
        q_start=q_start,
        q_len=torch.tensor(q_lens, device=dev, dtype=torch.int32),
        k_start=k_start,
        k_len=torch.tensor(k_lens, device=dev, dtype=torch.int32),
        max_q_len=max(q_lens),
        scale=scale,
        causal=causal,
        return_lse=True,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = _prefill_reference(
        q,
        k,
        v,
        q_start,
        torch.tensor(q_lens, device=dev, dtype=torch.int32),
        k_start,
        torch.tensor(k_lens, device=dev, dtype=torch.int32),
        scale,
        causal,
    )
    tol = 1e-2 if causal else 5e-3
    assert _rel(out.float(), ref_out) < tol
    assert float((lse - ref_lse).abs().max()) < 5e-2


class _Metadata:
    """The two attributes the backend reads off MLACommonPrefillMetadata."""

    def __init__(self, q_start_loc, max_query_len):
        self.query_start_loc = q_start_loc
        self.max_query_len = max_query_len


class _Chunk:
    def __init__(self, q_start_loc, cu_seq_lens, max_query_len, max_seq_len):
        self.query_start_loc = q_start_loc
        self.cu_seq_lens = cu_seq_lens
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len


def _backend():
    return Gfx906TritonMLAPrefill(
        num_heads=12,
        scale=1.0 / math.sqrt(256),
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        vllm_config=None,
    )


def test_backend_new_tokens_and_context_chunk():
    dev = torch.device("cuda:0")
    heads, dim = 12, 256
    scale = 1.0 / math.sqrt(dim)
    torch.manual_seed(3)
    backend = _backend()
    q = torch.randn(48, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    k = torch.randn(48, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(48, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    loc = torch.tensor([0, 30, 48], device=dev, dtype=torch.int32)
    backend.prepare_metadata(_Metadata(loc, 30))

    out, lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    ref_out, ref_lse = _prefill_reference(
        q,
        k,
        v,
        loc[:-1],
        torch.tensor([30, 18], device=dev, dtype=torch.int32),
        loc[:-1],
        torch.tensor([30, 18], device=dev, dtype=torch.int32),
        scale,
        True,
    )
    torch.cuda.synchronize()
    assert out.shape == q.shape and lse.shape == (48, heads)
    assert _rel(out.float(), ref_out) < 1e-2

    # out= is honoured in place (supports_out() is True: v is not padded)
    buf = torch.zeros_like(q)
    backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False, out=buf)
    torch.cuda.synchronize()
    assert torch.equal(buf, out)

    # context chunk: 32 cached rows attended by 20 query rows of the same request
    qc = torch.randn(20, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    kc = torch.randn(32, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    vc = torch.randn(32, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    chunk = _Chunk(
        q_start_loc=torch.tensor([0, 20], device=dev, dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 32], device=dev, dtype=torch.int32),
        max_query_len=20,
        max_seq_len=32,
    )
    cout, close = backend.run_prefill_context_chunk(chunk, qc, kc, vc)
    ref_c, ref_c_lse = _prefill_reference(
        qc,
        kc,
        vc,
        torch.tensor([0], device=dev, dtype=torch.int32),
        torch.tensor([20], device=dev, dtype=torch.int32),
        torch.tensor([0], device=dev, dtype=torch.int32),
        torch.tensor([32], device=dev, dtype=torch.int32),
        scale,
        False,
    )
    torch.cuda.synchronize()
    assert _rel(cout.float(), ref_c) < 5e-3
    assert float((close - ref_c_lse).abs().max()) < 5e-2
