# QWEN-MLA — the qwen-mla-vllm retrofit plugin, vendored and ported to gfx906 / vLLM 0.32

> Branch `main` (user decision: no topic branch) · model `TelperionAI/Qwen3.8-27B-GLA-g2`
> family (27B dense: 48 GDN + 16 MLA layers) · date `2026-10-23` · roadmap item: none yet.

**VERDICT:** `SHIPPED` (the port; serving works with prefix caching off, one open crash)

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

## First real-checkpoint bring-up (2026-10-23)

`Qwen3.8-27B-GLA-g2-INT4` on 2x Radeon Pro VII, TP=2, `--dtype float16`,
`--kv-cache-dtype float16`, text-only, 1-2 GiB KV cap:

- Weights load in 14-17 s; **10.6 GiB of weights per GPU** (22 GB / 2 + shared embedding
  and vision shards). That leaves ~5 GiB for activations + KV on a 16 GB card, which is
  why `--gpu-memory-utilization 0.92` OOM'd in the KV allocation (11.36 GiB torch-allocated,
  3.58 GiB free, 4.00 GiB requested): cap KV explicitly with `--kv-cache-memory-bytes`.
- Page unification does what the geometry predicts: vLLM raises the attention block size to
  cover the linear-attention state page (`Setting attention block size to 784/1024 tokens`,
  `Padding mamba page size by 0.13%`), so the three MLA row widths share one page.
- The rms invariant from the plugin's calibration blog holds: measured `rms` 1.30 / 1.39
  (documented range 0.29-2.1), which says the partial-RoPE / q-permutation / latent pairing
  produce sane keys.
- Output is **correct** in eager and in graph mode, with and without prefix caching, at
  `max-model-len` 1024: `' Paris.\nThe capital of Germany is Berlin.'`, `391` for 17*23, and
  an exact echo test.
- **Throughput** (graph mode, TP=2, 2 GiB KV, batch 1, `max-model-len` 2048, greedy):

  | arm | result |
  |---|---|
  | decode, short prompt | 14.5 tok/s (128 tokens, includes the prefill) |
  | decode, ~340-token context | 20.9 tok/s (64 tokens) |
  | prefill, 342-token prompt | **411 tok/s** (3 runs, 0.83 s each) |

  For reference the fork's base model (no retrofit) runs 40.8 tok/s TP=2 greedy at 2k on
  this box, so the retrofit is at roughly half the teacher's decode with MTP off and the
  generic Triton MLA kernel -- expected for a first cut, and the obvious next lever is
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
  These are single-sample numbers: read `dvfs-mi50.md` before quoting them.

Two defects surfaced, one fixed and one open:

1. **bf16 checkpoints need `--dtype float16` on this platform.** Serving bf16 produced
   `expected mat1 and mat2 to have the same dtype: Half != BFloat16`, then a
   `fused_add_rms_norm` assert, then `Both operands must be same dtype` in `tl.dot`: a
   linear's output dtype is the backend's choice here, and it does not always match the
   weights. The port now pins projection outputs and the prefill kernel's `q` to the layer
   input dtype, and `_k_up_raw` follows the served dtype after load.
2. **Partial prefix-cache hits kill the worker (OPEN).** With the plugin's
   `--prefix-match-unit 16` default (our patch), a second request sharing a prompt prefix
   gets a hit that lands *inside* a block (192 of 198 tokens). vLLM then issues a
   copy-on-write of 64 blocks and the TP1 worker dies with `exit code: None` (no traceback
   -- a hard fault, not a Python error). With `QWEN_MLA_PREFIX_UNIT=0` (partial hits off)
   the same traffic is clean (block-aligned hits: `1st/2nd/3rd/4th` requests, the third
   adding a suffix so the hit lands inside a block, all coherent, no crash). 0.32 already
   trims every full-attention group (the plugin's third prefix-cache fix is upstream), so
   the surviving suspect is 0.32's new block-copy path (`kv_cache_block_copies` in the
   scheduler output) for this hybrid layout, not the hit reconciliation. Repro: two
   `/v1/completions` calls whose prompts share a prefix, at `--prefix-match-unit 16`.
   **The port now defaults `QWEN_MLA_PREFIX_UNIT=0`** (no partial hits); set it to 16 only
   when debugging that path.

## Where this stands, and what is next

**Works today** (recipe: `_serve_qwen_mla.sh`): TP=2, fp16, text-only, KV capped
explicitly; correct greedy output, 14.5-20.9 tok/s decode, 411 tok/s prefill, 21.5k tokens
of KV at a 2 GiB cap, prefix caching limited to block-aligned hits.

**Open work is tracked in `ROADMAP.md`** (forward-looking material lives there, not here):

| id | item | why it matters |
|---|---|---|
| QMLA-1 | partial prefix-cache hits kill the TP worker | without it, hits align to `lcm(16,784,1024) = 50 176` tokens, i.e. no reuse below 50k; agent/multi-turn serving loses its main win |
| QMLA-2 | MTP k=3 | the teacher's drafter is unused; the plugin measured 1.3-2.1x, and half the teacher's decode speed is the gap to close |
| QMLA-3 | KV capacity on 16 GB cards | the vision tower and MTP weights cost ~1.5 GiB of KV for a text-only workload |
| QMLA-4 | decode/prefill tuning | the vendored kernel reaches ~20 GB/s of ~800 GB/s peak at the TP=2 local shapes |
| QMLA-5 | quality gate vs the teacher | nothing has been compared to `Qwen3.8-27B-AWQ-INT4` yet |
| QMLA-6 | recipe + docs | done for the recipe (`_serve_qwen_mla.sh`); README row pending |

Not attempted, still true: TP=1 (the widest latent needs the CUDA-gated two-pass kernel),
FP8 KV (no hardware conversion), vision serving (weights load, but nothing was exercised),
and DCP (refused by the metadata builder).

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
