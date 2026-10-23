# SPDX-License-Identifier: Apache-2.0
"""Runtime patches the vendored MLA plugin needs against vLLM 0.32.

Every entry here is a *plugin-side* workaround for something vLLM has no extension point
for. Nothing in vLLM's own files is edited, and each patch is scoped to our
architectures so stock models are untouched. Where a patch depends on an internal shape,
it asserts that shape and says so in the failure message: a merge that moves the ground
under a patch must fail loudly, not degrade silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

ARCHS: frozenset[str] = (
    frozenset()
)  # filled by gfx906_mla.__init__ before register() runs


def _log(msg: str) -> None:
    print(f"[gfx906-mla] {msg}", file=sys.stderr, flush=True)


def is_ours(archs) -> bool:
    return any(a in ARCHS for a in (archs or []))


def ours_from(vllm_config) -> bool:
    try:
        archs = vllm_config.model_config.architectures
    except Exception:  # noqa: BLE001 - config not resolved yet
        return False
    return is_ours(archs)


def alias_legacy_env() -> None:
    """Honour pre-rename MVLA_* environment variables as QWEN_MLA_* when unset."""
    for key, value in list(os.environ.items()):
        if key.startswith("MVLA_"):
            os.environ.setdefault("QWEN_MLA_" + key[len("MVLA_") :], value)


def alias_legacy_config_keys() -> None:
    """Mirror pre-rename mvla_* checkpoint config keys onto mla_*."""
    import vllm.config.model as _m

    if getattr(_m.get_config, "_gfx906_mla_aliased", False):
        return
    orig = _m.get_config

    def _mirror(cfg) -> None:
        if cfg is None:
            return
        for key in list(vars(cfg)):
            if key.startswith("mvla_") and not hasattr(
                cfg, "mla_" + key[len("mvla_") :]
            ):
                setattr(cfg, "mla_" + key[len("mvla_") :], getattr(cfg, key))

    def get_config(*args, **kwargs):
        cfg = orig(*args, **kwargs)
        _mirror(cfg)
        _mirror(getattr(cfg, "text_config", None))
        return cfg

    get_config._gfx906_mla_aliased = True
    _m.get_config = get_config


def fingerprint_compile_cache() -> None:
    """Fold the plugin's code and the checkpoint's mla_* config into vLLM's
    compile-cache key.

    ModelConfig.compute_hash covers the config fields but not the HF config contents or
    this package's source, so a torch.compile artifact built for another plugin version
    or another mla_* layout would be reused silently for the same model path.
    """
    import vllm.config.model as _m

    if getattr(_m.ModelConfig.compute_hash, "_gfx906_mla", False):
        return
    digest = hashlib.sha256()
    for path in sorted(pathlib.Path(__file__).resolve().parent.rglob("*.py")):
        digest.update(path.read_bytes())
    code_digest = digest.hexdigest()
    orig = _m.ModelConfig.compute_hash

    def compute_hash(self):
        base = orig(self)
        try:
            archs = list(getattr(self, "architectures", None) or [])
        except Exception:  # noqa: BLE001
            archs = []
        if not is_ours(archs):
            return base
        cfg = getattr(self, "hf_text_config", None) or getattr(self, "hf_config", None)
        keys = (
            {k: repr(v) for k, v in sorted(vars(cfg).items()) if k.startswith("mla_")}
            if cfg is not None
            else {}
        )
        extra = json.dumps(
            {"archs": archs, "mla": keys, "code": code_digest}, sort_keys=True
        )
        return hashlib.sha256((base + extra).encode()).hexdigest()

    compute_hash._gfx906_mla = True
    _m.ModelConfig.compute_hash = compute_hash


def inherit_qwen3_5_config_hook() -> None:
    """Give the plugin architectures the teacher's Qwen3.5 config hook.

    vLLM applies per-model config fixes by looking the ARCHITECTURE NAME up in
    MODELS_CONFIG_MAP. Qwen3.5's hook copies the checkpoint's ``mamba_ssm_dtype`` into
    the cache config; without it the 48 untouched linear-attention layers keep their
    recurrent state in a narrower dtype than the teacher, which has nothing to do with
    the retrofit.
    """
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )

    for arch in ARCHS:
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)


_ORIG_HF_CONFIG_OVERRIDE = None


def _mtp_hf_config_override(hf_config):
    """Module-level (not a closure): vLLM stores this callable in its config, and
    data-parallel
    startup pickles the config into each engine process."""
    archs = getattr(hf_config, "architectures", None) or []
    if getattr(hf_config, "model_type", None) == "qwen3_5_text" and is_ours(archs):
        hf_config.model_type = "qwen3_5_mtp"
        hf_config.update(
            {
                "n_predict": getattr(hf_config, "mtp_num_hidden_layers", None),
                "architectures": ["Qwen3_5MTP"],
            }
        )
        return hf_config
    return _ORIG_HF_CONFIG_OVERRIDE(hf_config)


def enable_mtp_draft() -> None:
    """Let ``--speculative-config '{"method": "mtp"}'`` find the Qwen3.5 MTP head.

    SpeculativeConfig.hf_config_override recognises Qwen3.5 only as the multimodal
    wrapper (model_type "qwen3_5"); text-only checkpoints are "qwen3_5_text", so without
    this the draft keeps the target's architecture and vLLM builds a second full model
    as the "draft".
    """
    global _ORIG_HF_CONFIG_OVERRIDE
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, "_gfx906_mla_mtp", False):
        return
    _ORIG_HF_CONFIG_OVERRIDE = SpeculativeConfig.hf_config_override
    SpeculativeConfig.hf_config_override = staticmethod(_mtp_hf_config_override)
    SpeculativeConfig._gfx906_mla_mtp = True


def spec_decode_graph_capture_fix() -> None:
    """Capture spec-decode graphs with the linear-attention layers in spec-decode mode.

    GDN's metadata builder tells speculative rows (1 + k tokens each) from prefills by
    ``num_decode_draft_tokens_cpu``, which the model runner fills from the scheduler --
    but only for real steps. Capture goes through ``_dummy_run``, which never touches
    the buffer, so it reads all zeros and classifies the uniform (1 + k)-token batch it
    is capturing as PREFILLS. The replayed graph then holds prefill-path kernels for
    what are spec-decode rows at run time.

    The converse case at run time: vLLM picks the FULL "uniform decode" graph from the
    batch SHAPE alone, so a prefill chunk of exactly 1 + k tokens replays that graph and
    GDN runs verification kernels on prompt tokens. Such batches are forced onto the
    mixed path.
    """
    import inspect

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner._dummy_run, "_gfx906_mla", False):
        return

    params = list(inspect.signature(GPUModelRunner._dummy_run).parameters)
    # `args` inside the wrapper starts after (self, num_tokens).
    uniform_idx = params.index("uniform_decode") - 2
    orig_dummy = GPUModelRunner._dummy_run

    def _dummy_run(self, num_tokens, *args, **kwargs):
        if (
            self.speculative_config is not None
            and hasattr(self, "num_decode_draft_tokens")
            and ours_from(self.vllm_config)
        ):
            uniform = kwargs.get(
                "uniform_decode",
                args[uniform_idx] if len(args) > uniform_idx else False,
            )
            qlen = self.uniform_decode_query_len
            self.num_decode_draft_tokens.np[:] = -1
            if uniform and qlen > 1:
                reqs = min(self.scheduler_config.max_num_seqs, -(-num_tokens // qlen))
                self.num_decode_draft_tokens.np[:reqs] = qlen - 1
                if num_tokens % qlen:
                    self.num_decode_draft_tokens.np[reqs - 1] = num_tokens % qlen - 1
                self.num_accepted_tokens.np[:reqs] = 1
                self.num_accepted_tokens.copy_to_gpu()
        return orig_dummy(self, num_tokens, *args, **kwargs)

    _dummy_run._gfx906_mla = True

    orig_prepare = GPUModelRunner._prepare_inputs
    orig_determine = GPUModelRunner._determine_batch_execution_and_padding

    def _prepare_inputs(self, scheduler_output, *args, **kwargs):
        if self.speculative_config is not None and ours_from(self.vllm_config):
            drafts = scheduler_output.scheduled_spec_decode_tokens
            self._gfx906_mla_all_spec = all(
                len(drafts.get(req, ())) + 1 == n
                for req, n in scheduler_output.num_scheduled_tokens.items()
            )
        return orig_prepare(self, scheduler_output, *args, **kwargs)

    def _determine_batch_execution_and_padding(*args, **kwargs):
        self = args[0]
        if (
            kwargs.get("force_uniform_decode") is None
            and getattr(self, "_gfx906_mla_all_spec", True) is False
        ):
            kwargs["force_uniform_decode"] = False
        self._gfx906_mla_all_spec = True
        return orig_determine(*args, **kwargs)

    _prepare_inputs._gfx906_mla = True
    _determine_batch_execution_and_padding._gfx906_mla = True
    GPUModelRunner._dummy_run = _dummy_run
    GPUModelRunner._prepare_inputs = _prepare_inputs
    GPUModelRunner._determine_batch_execution_and_padding = (
        _determine_batch_execution_and_padding
    )


def prefix_cache_hit_granularity() -> None:
    """Default ``--prefix-match-unit`` for prefix caching (``--mamba-cache-mode
    align``).

    The MLA layers have different latent widths, and vLLM equalises their page sizes by
    giving each width its own block size (1x, 2x, 4x the cache block). With the default
    hash unit (the GCD, which is also the linear-attention block) vLLM rounds every hit
    to the LCM of those blocks. A unit finer than the GDN block switches on vLLM's
    partial hits: full-attention groups match inside a block, and the scheduler ends one
    prefill chunk at the prompt's last unit boundary so the GDN state there is kept.
    """
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

    if getattr(
        HybridAttentionMambaModelConfig.verify_and_update_config, "_gfx906_mla", False
    ):
        return
    orig = HybridAttentionMambaModelConfig.verify_and_update_config.__func__

    def verify_and_update_config(cls, vllm_config):
        orig(cls, vllm_config)
        cache_config = vllm_config.cache_config
        if (
            ours_from(vllm_config)
            and cache_config.enable_prefix_caching
            and cache_config.mamba_cache_mode == "align"
            and cache_config.prefix_match_unit is None
        ):
            cache_config.prefix_match_unit = (
                16  # hybrid block sizes are multiples of 16
            )
            _log("prefix caching: hits land on 16-token boundaries")

    verify_and_update_config._gfx906_mla = True
    HybridAttentionMambaModelConfig.verify_and_update_config = classmethod(
        verify_and_update_config
    )


def profile_prefill_context_scratch() -> None:
    """Make the profile run see the whole per-chunk scratch of a long-context prefill.

    vLLM sizes the KV cache from the peak memory of a profile forward, in which every
    MLA layer stands in for the context-chunk path by allocating the kv_b_proj output
    alone: [workspace rows, heads, nope + v]. The real path holds more at once: the
    assembled key, the per-head rope selection feeding it, and the prefill backend's
    copy of v padded to the key width. Allocating those alongside the parent's stand-in
    makes the profiled peak cover the real one, so the KV budget is right by
    construction rather than by --kv-cache-memory-bytes.
    """
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    if getattr(MLAAttention.forward_impl, "_gfx906_mla", False):
        return

    orig = MLAAttention.forward_impl

    def forward_impl(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", args[4] if len(args) > 4 else None)
        if attn_metadata is not None or getattr(self.impl, "rms_offset", -1) < 0:
            return orig(self, *args, **kwargs)
        import torch

        latent = kwargs.get("k_c_normed", args[1] if len(args) > 1 else None)
        assert latent is not None, "MLA forward_impl without k_c_normed"
        rows, heads = self.chunked_prefill_workspace_size, self.num_heads
        kw = self.qk_nope_head_dim + getattr(
            self.impl, "mla_rope_dim", self.qk_rope_head_dim
        )
        dev, dtype = latent.device, latent.dtype
        held = [
            torch.empty((rows, heads, kw), device=dev, dtype=dtype),  # assembled key
            torch.empty(  # per-head rope selection
                (rows, heads, kw - self.qk_nope_head_dim), device=dev, dtype=dtype
            ),
            torch.empty(  # v padded to the key width
                (rows, heads, max(kw, self.v_head_dim)), device=dev, dtype=dtype
            ),
        ]
        # Margin for the query-side and indexing temporaries of the same call.
        sim = rows * heads * (self.qk_nope_head_dim + self.v_head_dim) * dtype.itemsize
        held.append(
            torch.empty(
                int(0.2 * (sim + sum(t.numel() * t.element_size() for t in held))),
                device=dev,
                dtype=torch.uint8,
            )
        )
        try:
            return orig(self, *args, **kwargs)
        finally:
            del held

    forward_impl._gfx906_mla = True
    MLAAttention.forward_impl = forward_impl


def skip_cudagraph_memory_profiling() -> None:
    """Keep the CUDA-graph memory estimate from overrunning on mixed padded page shapes.

    gpu_worker.determine_available_memory() calls profile_cudagraph_memory() whenever
    cudagraphs are enabled. That path allocates a minimal KV cache sized for UNPADDED
    pages and reshapes it through a stride that assumes the PADDED page stride; MLA pads
    its cache rows on purpose, so with more than one page shape the reshape overruns
    (``setStorage ... out of bounds``).

    Scoped to the env var vLLM already has for this: when the estimate would be
    discarded, do not compute it. A failure of the estimate itself is recovered the way
    vLLM's own no-graphs branch does -- release the profiling cache and report no
    estimate -- and only for that error.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_gfx906_mla_profiling", False):
        return
    GPUModelRunner._gfx906_mla_profiling = True

    if os.environ.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0":
        GPUModelRunner.profile_cudagraph_memory = lambda self: 0
        _log("cudagraph memory profiling skipped (requested by env)")
        return

    orig = GPUModelRunner.profile_cudagraph_memory

    def profile_cudagraph_memory(self) -> int:
        try:
            return orig(self)
        except RuntimeError as err:
            if "setStorage" not in str(err):
                raise
            self._cleanup_profiling_kv_cache()
            _log("cudagraph memory estimate skipped (mixed padded KV page shapes)")
            return 0

    GPUModelRunner.profile_cudagraph_memory = profile_cudagraph_memory


def force_bf16_kv_cache() -> None:
    """Resolve ``--kv-cache-dtype auto`` to bf16, not the checkpoint's FP8 default.

    The quantized checkpoints declare an 8-bit float kv_cache_scheme and carry
    calibrated scales, meaning "serve with an FP8 cache unless told otherwise". gfx906
    has no FP8 hardware conversion, so vLLM honouring that declaration would hand us a
    cache this port cannot run on. An explicit ``--kv-cache-dtype fp8`` still reaches
    the model and is refused there, with the flag to use instead.
    """
    import vllm.engine.arg_utils as arg_utils
    import vllm.utils.torch_utils as torch_utils

    if getattr(torch_utils, "_gfx906_mla_kv", False):
        return
    orig = torch_utils.resolve_kv_cache_dtype_string

    def resolve_kv_cache_dtype_string(kv_cache_dtype, model_config):
        resolved = orig(kv_cache_dtype, model_config)
        if kv_cache_dtype != "auto":
            return resolved
        if str(resolved).startswith("fp8"):
            archs = getattr(
                getattr(model_config, "hf_config", None), "architectures", None
            )
            if is_ours(archs):
                _log(
                    "the checkpoint's FP8 KV default is overridden to bfloat16 "
                    "(gfx906 has no FP8 conversion); pass --kv-cache-dtype bfloat16 "
                    "to silence this"
                )
                return "bfloat16"
        return resolved

    torch_utils.resolve_kv_cache_dtype_string = resolve_kv_cache_dtype_string
    arg_utils.resolve_kv_cache_dtype_string = resolve_kv_cache_dtype_string
    torch_utils._gfx906_mla_kv = True


def install(archs) -> None:
    """Apply every patch. Called once from register()."""
    global ARCHS
    ARCHS = frozenset(archs)
    alias_legacy_env()
    alias_legacy_config_keys()
    fingerprint_compile_cache()
    inherit_qwen3_5_config_hook()
    enable_mtp_draft()
    spec_decode_graph_capture_fix()
    prefix_cache_hit_granularity()
    profile_prefill_context_scratch()
    skip_cudagraph_memory_profiling()
    force_bf16_kv_cache()
