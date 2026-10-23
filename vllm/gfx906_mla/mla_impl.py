# SPDX-License-Identifier: Apache-2.0
"""MLA impl for the Qwen3.5-geometry retrofits on gfx906: absorbed decode + exact prefill k_norm.

WHAT DIFFERS FROM DeepSeek MLA, AND WHY IT NEEDS A SUBCLASS. DeepSeek normalises the LATENT
(kv_a_layernorm), so K stays linear in c and W_UK folds into the query with nothing left over.
Qwen3.5 applies k_norm to the ASSEMBLED [rope ; nope] key AFTER up-projection:

    q . k_norm(key_h) = ( [ (q_nope*(1+w)) @ W_UK_h ] . c  +  rope_term ) / rms(key_h)

The numerator absorbs exactly as usual. The denominator is a per-(key-token, HEAD) scalar and
the latent is shared across heads, so it cannot be folded into the cache:

  decode  : the Triton kernel divides the scores by rms, read from a tail appended to the KV
            cache (see ops/mla_decode_attention.py).
  prefill : k/v are materialised from the latent anyway, so k_norm is applied literally.

CACHE LAYOUT  [ latent | packed rope | rms (local heads) | page pad ]
The rms tail rides inside the declared qk_rope_head_dim so vLLM's own cache-write path moves it,
and paging / eviction / prefix caching move it for free. The decode kernel never reads it as
attention because Lk is pinned to rms_offset.

PREFILL RUNS AT THE STOCK (192, 64, 256) GEOMETRY: only decode needs the packed layout, so q and k
are narrowed at the two funnels below and the prefill backend sees a plain head-dim-256 problem
(see prefill.py).

Everything here is plugin-side: vLLM's own files are untouched, and each place where 0.32 needs a
different shape from 0.27.1 is handled by an override on our subclasses (or, where vLLM exposes no
hook, by a call-scoped substitution in this module).
"""

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)

from .ops.mla_decode_attention import decode_attention_fwd as mla_decode_attention_fwd
from .ops.mla_gather import gather_cache_torch

FP8_RMS_LOG2_LO, FP8_RMS_LOG2_HI = -3.0, 3.0


def device_memory_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / 2**30


def chunked_workspace_rows(vllm_config) -> int:
    """Rows of the chunked-prefill context workspace: vLLM's rule without its page floor.

    vLLM sizes it min(max(8 x max_model_len, 4 x max_num_seqs x page), 64k) and then floors it at
    max_num_seqs x page, so every prefill can get a page-aligned chunk. Our rows are wider than
    DeepSeek's and the floor costs 3.3 GB of workspace at the default 1024 sequences. Our gather
    handles chunk starts at any token, so build() drops the page alignment and one row per prefill
    is the real floor.
    """
    sched, model = vllm_config.scheduler_config, vllm_config.model_config
    forced = os.environ.get("QWEN_MLA_PREFILL_WS_ROWS")
    if forced:
        return max(int(forced), sched.max_num_seqs)
    rows = min(8 * model.max_model_len, 64 * 1024)
    # Each row costs ~30 KB of per-chunk scratch at prefill; cap at 1024 rows per GiB of device
    # memory (15k rows on a 16 GB card, unchanged at 64k from 64 GB up).
    gib = device_memory_gib()
    if gib:
        rows = min(rows, max(4096, int(gib) * 1024))
    return max(rows, sched.max_num_seqs)


def splitk_cap(batch: int, heads: int, lse_dim: int, splits: int) -> int:
    """Largest split count <= `splits` whose fp32 split-K scratch fits the budget.

    vLLM sizes the Triton decode accumulator for the worst case (max_num_seqs, and splits from
    max_model_len) before KV allocation: with defaults on a large part and a 1792-wide latent that
    is tens of GB. Halving the splits only trades parallelism over keys, and the measured
    configurations stay at their natural split count under the default budget.
    """
    budget = int(float(os.environ.get("QWEN_MLA_SPLITK_SCRATCH_MB", 512)) * (1 << 20))
    while splits > 1 and batch * heads * splits * lse_dim * 4 > budget:
        splits //= 2
    return max(1, splits)


def splitk_reserve_bytes(batch: int, heads: int, lse_dim: int, splits: int) -> int:
    """Bytes that cover every splitk_cap()-ed request with batch <= `batch`."""
    budget = int(float(os.environ.get("QWEN_MLA_SPLITK_SCRATCH_MB", 512)) * (1 << 20))
    full = batch * heads * splits * lse_dim * 4
    return min(full, max(budget, batch * heads * lse_dim * 4))


def mla_kv_fp8(vllm_config=None) -> bool:
    """Whether the KV cache is an FP8 one.

    gfx906 has no FP8 conversion instructions, so an FP8 cache is not a supported configuration
    here: the quantized checkpoints declare one by default and vLLM would honour that, so refuse
    it explicitly instead of running the emulated path at a fraction of the speed.
    """
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except Exception:  # noqa: BLE001 - outside a config context: assume bf16
            return False
    cache_dtype = str(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
    if cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "gfx906 has no FP8 hardware conversion, so this port supports bf16 (or fp16) KV "
            "caches only. Pass --kv-cache-dtype bfloat16."
        )
    return False


# rms tail formats for an FP8 cache (kept so the kernels' constexprs stay reachable):
#   log8 (default): one byte per head, uint8 code of log2(rms) uniform over [LO, HI]
#   bf16:           two bytes per head (raw bf16)
#   e4m3:           one byte per head, scaled with the row like latent and rope
def mla_fp8_rms_mode() -> str:
    mode = os.environ.get("QWEN_MLA_FP8_RMS", "log8")
    if mode not in ("log8", "bf16", "e4m3"):
        raise ValueError(f"QWEN_MLA_FP8_RMS={mode!r}: expected log8, bf16 or e4m3")
    return mode


def mla_fp8_rms_bf16() -> bool:
    """True for the wide FP8 row (raw bf16 rms, two bytes per head)."""
    return mla_fp8_rms_mode() == "bf16"


def mla_rope_span(hf, tp_size: int = 1) -> int:
    """How many rope GROUPS a rank's heads span -- i.e. how many are worth caching.

    k_rope_proj is replicated so every rank computes all four, but a rank only reads the groups
    its own query heads belong to: at TP=2 rank 0 spans groups 0-1 and rank 1 spans 2-3.
    """
    group = hf.num_attention_heads // hf.num_key_value_heads
    n_local = hf.num_attention_heads // tp_size
    return max(
        (r * n_local + n_local - 1) // group - (r * n_local) // group + 1
        for r in range(tp_size)
    )


def mla_base_tail(hf, tp_size: int = 1, fp8: bool = False) -> int:
    """Live cache tail beyond the latent: packed rope (4x64) + one rms scalar per LOCAL head."""
    from .partial_rope import effective_rope_dim

    rope = effective_rope_dim(hf)
    per_head = 2 if (fp8 and mla_fp8_rms_bf16()) else 1
    return (
        mla_rope_span(hf, tp_size) * rope
        + (hf.num_attention_heads // tp_size) * per_head
    )


def mla_gla_split(hf, tp_size: int) -> int:
    """Latent groups each TP rank's slice is divided by: G when the GLA groups align with TP.

    A GLA checkpoint (mla_latent_groups = G > 1) has block-diagonal up-projections: query heads of
    group g read only latent columns [g*r/G, (g+1)*r/G). When tp_size is a multiple of G every rank's
    heads lie in ONE group, so the rank needs -- and caches -- only that group's r/G latent dims.
    """
    g = int(getattr(hf, "mla_latent_groups", 1) or 1)
    if g > 1 and tp_size % g == 0 and os.environ.get("QWEN_MLA_GLA_SHARD", "1") != "0":
        return g
    return 1


def mla_local_ranks(hf, tp_size: int = 1) -> dict[int, int]:
    """Per-layer latent width as CACHED on one TP rank (see mla_gla_split)."""
    g = mla_gla_split(hf, tp_size)
    return {int(k): int(v) // g for k, v in hf.mla_ranks.items()}


def mla_padded_head_size(
    hf, kv_lora_rank: int, tp_size: int = 1, fp8: bool = False
) -> int:
    """Cache row width for a layer, padded so every layer's PAGE divides the largest.

    vLLM unifies differing page sizes by scaling block_size -- but only when the largest page is
    an exact multiple of the others. Our allocation gives rows of 536 / 1048 / 2072, whose ratios
    are 67:131:259, so nothing divides and ALL the full-attention layers get padded to the widest
    row, collapsing the compression the checkpoint was trained for. Rounding each row up to a
    power-of-two MULTIPLE of the smallest row makes the ratios 1:2:4, so unification is by block
    size and the only cost is the rounding itself.

    The padding sits past the rms tail and is never read: the decode kernel bounds its attention
    reads at rms_offset, and prefill slices rope/rms at explicit offsets.

    ``mla_row_scheme`` in the checkpoint config pins the scheme ("pow2" or "mult") when the
    default choice cannot be reconciled with the model's other page sizes. vLLM 0.32 refuses to
    pad an MLA page to the maximum (its rows are indexed whole), so a row whose page does not
    divide the widest page on the rank -- the linear-attention state page, in these hybrid
    models -- fails unification outright; the scheme is then the only knob the plugin has.
    """
    tail = mla_base_tail(hf, tp_size, fp8)
    counts: dict[int, int] = {}
    for v in mla_local_ranks(hf, tp_size).values():
        counts[int(v)] = counts.get(int(v), 0) + 1
    ranks = sorted(counts)
    base = ((min(r + tail for r in ranks) + 15) // 16) * 16

    def _pow2(x: int) -> int:
        w = 1
        while w < x:
            w *= 2
        return w

    schemes = {
        "pow2": {r: _pow2(r + tail) for r in ranks},
        "mult": {r: base * _pow2(-(-(r + tail) // base)) for r in ranks},
    }
    pinned = getattr(hf, "mla_row_scheme", None)
    if pinned not in (None, "auto") and pinned not in schemes:
        raise ValueError(
            f"mla_row_scheme={pinned!r}: expected 'auto', 'pow2' or 'mult'"
        )
    candidates = []
    for name, rows in schemes.items():
        if pinned not in (None, "auto") and name != pinned:
            continue
        widest = max(rows.values())
        if any(widest % v for v in rows.values()):
            continue  # pages would not unify
        if any(v % 16 for v in rows.values()):
            continue  # 16-element alignment for the kernel's tile math
        candidates.append(
            (sum(rows[r] * counts[r] for r in ranks) / sum(counts.values()), rows)
        )
    if candidates:
        rows = min(candidates, key=lambda t: t[0])[1]
        if kv_lora_rank in rows:
            return rows[kv_lora_rank]
    return _pow2(kv_lora_rank + tail)


def mla_rank_by_head_size(hf, tp_size: int = 1, fp8: bool = False) -> dict[int, int]:
    """Padded head_size -> kv_lora_rank. Max on collision: the value sizes a workspace."""
    out: dict[int, int] = {}
    for v in mla_local_ranks(hf, tp_size).values():
        rank = int(v)
        out[mla_padded_head_size(hf, rank, tp_size, fp8)] = max(
            out.get(mla_padded_head_size(hf, rank, tp_size, fp8), 0), rank
        )
    return out


class QwenMLATritonImpl(TritonMLAImpl):
    """TritonMLAImpl with the rms divisor wired into decode and k_norm into prefill."""

    # set by the model definition before the first forward
    rms_offset: int = -1
    mla_local_heads: int = 0  # rms scalars in the k_pe tail = this rank's LOCAL heads
    mla_rope_dim: int = 0  # per-head rope width: 64, or 58 under partial rope
    mla_rope_group: int = 0  # query heads per KV head (6)
    mla_head_offset: int = 0  # this rank's first GLOBAL head index
    mla_rope_first: int = 0  # first GLOBAL rope group this rank caches
    mla_packed_rope: int = 0  # width of the packed rope block, EXCLUDING rms + pad

    def _mla_kernel(self, *args, **kwargs):
        kwargs["rms_offset"] = self.rms_offset
        kwargs["qrows"] = 1
        kwargs["rms_bf16"] = False
        kwargs["rms_log8"] = False
        return mla_decode_attention_fwd(*args, **kwargs)

    def _mla_group_of_local_head(self, n_local: int, device) -> torch.Tensor:
        """Rope group owned by each rank-local head, indexed relative to this rank's window."""
        gh = self.mla_head_offset + torch.arange(n_local, device=device)
        return gh // self.mla_rope_group - self.mla_rope_first

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Assemble the PREFILL key at stock width: [ k_nope | this head's rope ] / rms.

        k_pe arrives from the cache as [ packed rope | rms ] because the rms tail rides inside the
        declared qk_rope_head_dim, so prefill gets the divisor for free.

        (1+w) is NOT applied here: it is already folded into kv_b_proj's K rows, so k_nope arrives
        pre-scaled. Applying it again would square it. Only the division by rms remains.
        """
        if self.rms_offset < 0:
            return super()._concat_k_nope_k_pe(k_nope, k_pe)

        t, n_local = k_nope.shape[0], k_nope.shape[-2]
        # Slice at EXPLICIT offsets, never from the end: the row is padded past the rms tail so
        # that page sizes unify by block size, and measuring back from k_pe.shape[-1] would read
        # padding as rms.
        n_rope = self.mla_packed_rope
        flat = k_pe.reshape(t, -1)
        groups = self._mla_group_of_local_head(n_local, k_nope.device)

        rope = flat[:, :n_rope].view(t, -1, self.mla_rope_dim)[:, groups, :]
        rms = flat[:, n_rope : n_rope + n_local]

        k = torch.empty(
            (t, n_local, k_nope.shape[-1] + self.mla_rope_dim),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = rope.to(k_nope.dtype)
        # In place: an out-of-place divide holds a second full key alongside the first.
        return k.div_(rms.unsqueeze(-1).to(k.dtype).clamp(min=1e-6))

    def _mla_narrow_q(self, q: torch.Tensor) -> torch.Tensor:
        """Narrow q from the packed rope to this head's own rope slice.

        q is [ nope | packed rope | rms slot ]. Every slot outside the head's own rope group is
        zero by construction, so this drops nothing -- it just stops handing the backend hundreds
        of dims of which most are structurally zero.
        """
        t, n_local = q.shape[0], q.shape[-2]
        nope = q[..., : self.qk_nope_head_dim]
        packed = q[
            ..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.mla_packed_rope
        ]
        groups = self._mla_group_of_local_head(n_local, q.device)
        heads = torch.arange(n_local, device=q.device)
        sel = packed.view(t, n_local, -1, self.mla_rope_dim)[:, heads, groups, :]
        return torch.cat([nope, sel], dim=-1)

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        if self.rms_offset < 0:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        q = self._mla_narrow_q(q)
        # The chunked-CONTEXT gather is a CUDA kernel that hard-rejects entry sizes outside
        # DeepSeek's 320/576, so every prefill with cached context dies on our rows. Swap in the
        # torch equivalent for the duration of the call. Patch the module attribute, not a local
        # import: the parent resolves `ops.gather_and_maybe_dequant_cache` at call time.
        import vllm._custom_ops as _ops

        orig = _ops.gather_and_maybe_dequant_cache
        _ops.gather_and_maybe_dequant_cache = gather_cache_torch
        try:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        finally:
            _ops.gather_and_maybe_dequant_cache = orig

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        """Decode with the rms divisor: swap in the fork's kernel for the duration of the call.

        The parent resolves both names at call time, so patching the module attribute (not
        re-importing) is what makes the substitution take effect.
        """
        import vllm.v1.attention.backends.mla.triton_mla as _t

        q0 = q[0] if isinstance(q, tuple) else q
        batch, heads, lse_dim = q0.shape[0], q0.shape[1], self.kv_lora_rank + 1
        orig, orig_splits = _t.decode_attention_fwd, _t._compute_num_kv_splits
        _t.decode_attention_fwd = self._mla_kernel
        _t._compute_num_kv_splits = lambda max_seq_len, sm_count: splitk_cap(
            batch, heads, lse_dim, orig_splits(max_seq_len, sm_count)
        )
        try:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        finally:
            _t.decode_attention_fwd, _t._compute_num_kv_splits = orig, orig_splits


class MLAMetadata(MLACommonMetadata):
    """MLACommonMetadata without the DeepSeek-only head-size guard.

    __post_init__ rejects any head_dim outside [320, 576] -- DeepSeek's 256+64 and 512+64. Our
    cache rows are kv_lora_rank + tail, i.e. 536 / 1048 / 2072 across the allocation, so the guard
    fires on every layer, and it is the field's ONLY consumer.
    """

    def __post_init__(self):
        return


class MLATritonMetadataBuilder(TritonMLAMetadataBuilder):
    """Per-layer latent ranks and a right-sized chunked-prefill workspace.

    0.32 takes `mla_dims` from the attention layer, so the per-layer rank needs no config surgery;
    what still needs overriding is the workspace size, which vLLM derives from page-aligned chunks
    and caps in a way that costs gigabytes here.
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    query_len_support = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        self.metadata_cls = MLAMetadata

        rows = chunked_workspace_rows(vllm_config)
        if rows != self.chunked_prefill_workspace_size:
            width = self.chunked_prefill_workspace.shape[-1]
            dtype, dev = (
                self.chunked_prefill_workspace.dtype,
                self.chunked_prefill_workspace.device,
            )
            self.chunked_prefill_workspace = None
            self.chunked_prefill_workspace_size = rows
            self.chunked_prefill_workspace = torch.empty(
                (rows, width), dtype=dtype, device=dev
            )

        # ---- decode context parallelism ------------------------------------------------
        # DCP shards the cache by token and all-gathers q, so every rank must serve all GLOBAL
        # heads for the tokens it owns. Our cache row cannot: it stores only the rope groups this
        # TP rank's heads span and one rms slot per LOCAL head. Refuse it rather than serve
        # numerically wrong output that still looks fluent.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "MLA does not support decode context parallelism yet. DCP requires every rank "
                "to serve all global heads for its token shard, but the cache row stores only "
                "this rank's rope window and its local heads' rms."
            )

    def _reserve_attn_logits_workspace(self) -> None:
        # The parent reserves the split-K accumulator at max_num_seqs x max-context splits; see
        # splitk_cap for why that cannot stand here.
        from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            return
        batch = (
            self.vllm_config.scheduler_config.max_num_seqs
            * self.reorder_batch_threshold
        )
        heads = self.num_heads * self.dcp_world_size
        splits = _compute_num_kv_splits(
            self.model_config.max_model_len, current_platform.num_compute_units()
        )
        n = splitk_reserve_bytes(batch, heads, self.mla_dims.kv_lora_rank + 1, splits)
        current_workspace_manager().get_simultaneous(((n // 4,), torch.float32))

    def build(self, *args, **kwargs):
        # Unaligned context chunks (see chunked_workspace_rows); the parent hard-codes alignment.
        import vllm.model_executor.layers.attention.mla_attention as _m

        orig = _m.build_mla_chunked_context_metadata
        _m.build_mla_chunked_context_metadata = lambda **kw: orig(
            **{**kw, "align_chunk_to_block": False}
        )
        try:
            return super().build(*args, **kwargs)
        finally:
            _m.build_mla_chunked_context_metadata = orig


class QwenMLATritonBackend(TritonMLABackend):
    @staticmethod
    def get_name() -> str:
        return "QWEN_MLA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[QwenMLATritonImpl]:
        return QwenMLATritonImpl

    @staticmethod
    def get_builder_cls() -> type[MLATritonMetadataBuilder]:
        return MLATritonMetadataBuilder
