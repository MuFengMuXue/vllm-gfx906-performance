# QWEN-MLA — the qwen-mla-vllm retrofit plugin, vendored and ported to gfx906 / vLLM 0.32

> Branch `main` (user decision: no topic branch) · model `TelperionAI/Qwen3.8-27B-GLA-g2`
> family (27B dense: 48 GDN + 16 MLA layers) · date `2026-10-23` · roadmap item: none yet.

**VERDICT:** `SHIPPED` (the port; the *model* gate is still outstanding)

**GATE:** `tests/gfx906_mla/` — decode-kernel and prefill-kernel numerics, then a tiny
2-layer MLA model generating end to end (`pytest tests/gfx906_mla`). The real serving gate,
wall-clock on `Qwen3.8-27B-GLA-g2-INT4` @ TP=2, is not reachable from this box yet (the
checkpoint is 23 GB and Hugging Face is unreachable), so no serving number is claimed
here.

---

## HYPOTHESIS

The plugin's fast decode path (FlashInfer's patched CUDA kernel) is NVIDIA-only, but its
fallback is a Triton MLA decode kernel that already carries `is_hip_` branches, and its
prefill path delegates to vLLM's platform-selected MLA prefill backend. If the Triton
decode kernel reproduces reference numerics on gfx906, then the retrofit can be served
here once (a) the plugin is adapted to vLLM 0.32 and (b) a gfx906 MLA prefill backend
exists — with the CUDA-only pieces dropped rather than ported.

## What was done

- Vendored the plugin (Apache-2.0, TelperionAI/sootaugur, v0.6.3) into
  `vllm/gfx906_mla/` as a fork-only package, registered through the
  `vllm.general_plugins` entry point (`pyproject.toml`), next to `gfx906_fa`.
- Dropped the CUDA-only half: `fi_decode.py`, `fi_mla.py`, `kernels/*.cuh`,
  `patch_mla.py` (FlashInfer JIT overlay), `dcp_fix.py`, `prefill_align.py` (trtllm-gen /
  FA4 TMA alignment), the materialised and NoPE research variants, and the FP8 KV path
  (gfx906 has no FP8 conversion; `mla_kv_fp8` now refuses it with the flag to use instead).
- Kept verbatim: `ops/mla_decode_attention.py` (rms-aware MLA decode) and
  `ops/mla_gather.py` (torch fallback for the 320/576-only context gather; FP8 branches
  removed).
- New `ops/prefill_attention.py` + `prefill.py`: a varlen Triton flash attention with
  separate q/k offsets and an LSE output, exposed as `Gfx906TritonMLAPrefill` and injected
  with `MLAAttention(prefill_backend_cls=...)` — vLLM ships no MLA prefill backend for
  this device (`get_mla_prefill_backend_cls` raises: AITER FA needs gfx90a+, flash-attn
  is unavailable).
- 0.32 API deltas absorbed in the plugin layer (no vLLM file edited):
  `_build_decode` gained `max_query_len` (the plugin's flattening override is dropped --
  0.32 flattens in `forward_mqa`); `forward_mha` gained `k_scale`/`output`/`output_scale`;
  the metadata builder now reads `mla_dims` off the layer, so the per-layer latent-rank
  config surgery went away; the top-level weight mapper had to stop carrying the
  retrofit's stacked map (0.32 maps at both levels, which turned `.in_proj_b` inside
  `.in_proj_ba` into `in_proj_baa`); `CompilationConfig` has no `level` field.
- `patches.py` keeps only what has no extension point: legacy env/config aliases, the
  compile-cache fingerprint, the Qwen3.5 config hook, MTP draft discovery, the
  spec-decode graph-capture fix (GDN dummy-run draft tokens), `--prefix-match-unit` for
  hybrid prefix caching, the profile-run scratch estimate, the cudagraph memory-estimate
  guard, and the bf16 KV resolution (checkpoints declare FP8).
- `pyproject.toml`: entry point + per-file `E501` exemptions for the vendored kernel and
  the plugin's prose.

## The real checkpoint's geometry (T2=2 target)

`Qwen3.8-27B-GLA-g2-INT4` (22 GB, compressed-tensors `pack-quantized` W4A16, latent
projections excluded from quantization):

- 64 layers, full attention every 4th (`layer_types`), 24 query heads / 4 KV heads,
  `head_dim` 256, hidden 5120, vocab 248320, one MTP layer, vision tower present.
- `mla_ranks` = {256, 768, 1792} over 16 full-attention layers; `mla_latent_groups` = 2,
  so at TP=2 each rank caches **128 / 384 / 896** latent dims.
- Partial RoPE: `mla_rope_keep` keeps 29 of 32 subspaces → `rope_dim` 58
  (`mla_qk_rope_head_dim` 58), so the packed rope block is `2 groups * 58 = 116` and the
  rms tail is 12 → tail 128.
- Therefore rows are **256 / 512 / 1024** elements: an exact 1:2:4 ratio, so the page
  unifier scales block sizes instead of padding (which vLLM 0.32 refuses for MLA).
- The checkpoint declares an FP8 `kv_cache_scheme`; the port resolves `auto` to a
  non-FP8 cache and refuses an explicit `--kv-cache-dtype fp8`.

## Evidence — FOR

- Decode kernel numerics on MI50 (rank 512 / packed rope 128 / 24 heads, page 64, 900
  tokens): `cos 1.00000`, max abs error `3.4e-4`, LSE error `4.8e-7` versus an explicit
  torch reference. Gate: `tests/gfx906_mla/test_mla_kernels.py`.
- Prefill kernel numerics (h=12, d=256): causal `2.5e-3`, non-causal `6e-4`, cached-context
  chunk (q shorter than k) `3e-4`, LSE exact; `out=` path bit-identical.
- Decode microbench (page 784, 8k context, GLA-g2 TP=2 shapes): rank 448 at batch 8
  `7.41 ms/call` (~0.46 ms/layer), rank 896 at batch 1 `8.98 ms/call` (~0.56 ms/layer).
  **Launch-regime evidence**, not the gate: ~20 GB/s effective against ~800 GB/s peak, so
  there is room, but nothing here says what serving does.
- Page unification: the MLA page divides the model's linear-attention state page, so vLLM
  scales the MLA block size instead of padding (which 0.32 refuses for MLA). Verified on a
  tiny hybrid checkpoint: `kv cache group sizes (16, 32)`, `kv lcm block sizes 32`.
- **End to end on MI50**: the tiny pure-MLA checkpoint (2 layers, rank 64) generates
  through the registered entry point, our `QWEN_MLA_TRITON` decode backend and the new
  `GFX906_TRITON_MLA` prefill backend, exit 0, two distinct completions:

  ```
  [1..8]  -> [5, 104, 125, 76, 79, 113, 116, 80]
  [9,10,11] -> [38, 81, 80, 123, 99, 65, 79, 113]
  ```

  Random weights, so the tokens mean nothing; what the gate proves is that the cache row
  layout, the decode kernel, the prefill backend, the chunked-context merge and the hybrid
  page sizing all run together. Gate: `pytest tests/gfx906_mla`.

## Evidence — AGAINST

- **The widest latent does not fit the single-pass kernel on gfx906.** rank 1792 needs
  `BLOCK_DMODEL 2048` and the kernel asks for 128 KB of LDS against the device's 64 KB:
  `OutOfResources: shared memory, Required: 131072, Hardware limit: 65536`. The plugin's
  two-pass kernel is written for that shape but is gated `not is_hip_`, so a TP=1
  deployment of the widest layers is not servable as-is. TP=2 with the GLA-g2 checkpoints
  shards the latent to 896 and fits.
- **bf16 does not work on this platform; fp16 does.** Two separate failures pinned it
  down: `expected mat1 and mat2 to have the same dtype: Half != BFloat16` (a bf16-model
  linear returning fp16 for the small GEMM) followed by `fused_add_rms_norm ...
  layernorm_kernels.hip:320` once the residual stream met an fp16 projection output, and
  `Both operands must be same dtype. Got bf16 and fp16` inside `tl.dot` in the prefill
  kernel. Dtype pinning in the plugin layer (projections and outputs to the layer's input
  dtype; q follows k in the prefill kernel) makes the port robust, but serving a bf16
  *checkpoint* still needs `--dtype float16` on this fork, the same requirement the 27B
  AWQ recipe already carries. Recorded because it is the first thing the next bf16
  checkpoint will hit.
- The hybrid tiny checkpoint is slow to first token for reasons unrelated to this port:
  the first forward autotunes the stock Flash-Linear-Attention kernels (minutes on
  gfx906). The fast gate is therefore the pure-MLA variant; the hybrid one is
  `slow_test`.

## Interactions / superseded-by

- Replaces nothing; the fork had no dense-MLA serving path (the MLA-sparse work is
  MiniMax-M3/GLM, AITER-gated).
- The gfx906 Triton decode kernel the plugin forks is the same family as
  `vllm/v1/attention/ops/triton_decode_attention.py`, which already carries `on_gfx906()`
  launch tuning — a later step is to fold the rms support into that copy instead of
  keeping two forks.

## Refrigerated residue

- FP8 KV cache: the checkpoints ship calibrated scales and the kernel fork has the
  layout, but gfx906 has no FP8 conversion, so the path is refused rather than emulated.
  Reopen only if a software FP8 KV path measured cheap enough to pay for 2x capacity.
- Materialised (non-latent) MLA variant: kept out of the port, but it is the cheap way to
  separate "model plumbing is wrong" from "kernel is wrong" on a new box.
- The two-pass decode kernel looks like a ~1-day port for TP=1 `-MLA-INT4` on a 32 GB card
  (it needs a 64 KB-LDS-aware `DC`/`DVC`, not a rewrite).

## Search keys

`HYPOTHESIS:` MLA retrofit plugin is portable to gfx906 with a plugin-side prefill backend.
`VERDICT:` ported, kernel gates green; serving gate outstanding.
`gfx906_mla` `qwen-mla` `MLA` `prefill backend` `latent KV`.
