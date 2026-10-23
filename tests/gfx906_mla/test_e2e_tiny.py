# SPDX-License-Identifier: Apache-2.0
"""End-to-end gates for the vendored MLA port on a tiny synthetic checkpoint.

These are the only gates that exercise the whole stack at once: model registration
through the ``vllm.general_plugins`` entry point, the MLA layer's cache row layout, the
decode kernel, the prefill backend, and (in the hybrid case) page unification against
the linear-attention state page. The numeric kernels are gated separately in
``test_mla_kernels.py``; these only have to run and produce plausible output.

Requires the package entry point to be installed (``uv pip install -e .``), because vLLM
loads plugins from installed metadata in every worker process.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.gfx906_mla.tiny_mla_checkpoint import build

pytestmark = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a GPU"
)

VOCAB = 128


def _generate(model_dir: pathlib.Path, prompts=None, **llm_kwargs):
    import vllm.gfx906_mla  # noqa: F401  (registers the architectures in this process)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        max_model_len=256,
        gpu_memory_utilization=0.30,
        enforce_eager=True,
        dtype="float16",  # what this platform serves dense Qwen3.x in
        max_num_seqs=4,
        compilation_config={"mode": "none"},  # eager: compile time is not the gate
        **llm_kwargs,
    )
    if prompts is None:
        prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]]
    return llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))


def _check(out) -> None:
    assert len(out) == 2
    for result in out:
        # Random weights, so the tokens are arbitrary -- but they must exist, be in
        # range, and the two prompts must not collapse to one completion: a shape or
        # indexing bug in the cache row shows up as every request decoding the same.
        assert len(result.outputs[0].token_ids) == 8
        assert all(0 <= t < VOCAB for t in result.outputs[0].token_ids)
    assert out[0].outputs[0].token_ids != out[1].outputs[0].token_ids


def test_tiny_mla_model_generates(tmp_path: pathlib.Path) -> None:
    """Pure-MLA variant: the retrofit's own layers, with none of the teacher's GDN."""
    _check(_generate(build(tmp_path / "tiny_mla", hybrid=False)))


@pytest.mark.slow_test
def test_tiny_hybrid_model_generates(tmp_path: pathlib.Path) -> None:
    """Released-checkpoint shape: GDN layers next to the MLA ones.

    The first forward autotunes the stock Flash-Linear-Attention kernels (minutes on
    gfx906, unrelated to this port), hence the slow marker.
    """
    _check(_generate(build(tmp_path / "tiny_hybrid", hybrid=True)))


@pytest.mark.slow_test
def test_tiny_chunked_prefill_merges(tmp_path: pathlib.Path) -> None:
    """Prefill with cached context: chunked at 16 tokens, then a prefix-cache hit.

    The MLA context merge only runs when a prefill has cached context -- a chunk
    boundary, or a prefix hit -- and that is the path that faulted the served
    checkpoint while the plain single-chunk prefill was fine.
    """
    long_a = list(range(1, 41))
    long_b = list(range(60, 100))
    out = _generate(
        build(tmp_path / "tiny_mla_chunked", hybrid=False),
        [long_a, long_b, long_a],
        max_num_batched_tokens=16,
    )
    _check(out[:2])
    # The third request reuses the first request's KV; greedy output must match.
    assert out[2].outputs[0].token_ids == out[0].outputs[0].token_ids
