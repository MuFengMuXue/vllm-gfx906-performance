# SPDX-License-Identifier: Apache-2.0
"""gfx906 serving for the Qwen3.5-geometry MLA retrofits (vLLM 0.32 fork).

REGISTERED VIA ENTRY POINT, NOT A DIRECT CALL. vLLM runs the engine in a separate
process, so a ``ModelRegistry.register_model`` executed in the parent never reaches the
worker -- the model then silently resolves to a stock architecture and the run
"succeeds" while testing nothing. The ``vllm.general_plugins`` entry point is loaded in
every process, which is the only way the registration is actually in effect where the
model is built.

What this package is: the qwen-mla-vllm plugin (Apache-2.0, TelperionAI/sootaugur)
vendored into the fork and adapted to vLLM 0.32 and to gfx906. The CUDA-only fast path
(FlashInfer's JIT MLA kernel and its patched headers) is dropped; decode runs on the
Triton kernel in ``ops/mla_decode_attention.py``, and prefill on the new
``Gfx906TritonMLAPrefill`` because vLLM ships no MLA prefill backend for this device.
"""

from __future__ import annotations

import os
import sys

from . import patches

ARCHS = {
    # Serving path: latent KV cache, absorbed decode.
    "Qwen3_5MLAForCausalLM": "vllm.gfx906_mla.model:Qwen3_5MLAForCausalLM",
    # Same, behind the base model's vision tower.
    "Qwen3_5MLAForConditionalGeneration": (
        "vllm.gfx906_mla.model_vl:Qwen3_5MLAForConditionalGeneration"
    ),
}
LEGACY_ARCHS = {
    "Qwen3_5MVLAAbsorbedForCausalLM": "Qwen3_5MLAForCausalLM",
}

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:  # the entry point may be loaded more than once per process
        return
    _REGISTERED = True

    from vllm import ModelRegistry

    print(
        f"[gfx906-mla] register() called in pid {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )

    patches.install(ARCHS)
    for name, target in ARCHS.items():
        ModelRegistry.register_model(name, target)
    for legacy, canonical in LEGACY_ARCHS.items():
        ModelRegistry.register_model(legacy, ARCHS[canonical])
