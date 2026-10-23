# SPDX-License-Identifier: Apache-2.0
"""Build a tiny Qwen3.5+MLA retrofit checkpoint, for smoke-testing this port offline.

The real checkpoints are 23-56 GB and are not reachable from a build box, so the same
code paths run on a 2-layer model: one Gated DeltaNet layer (the teacher's, untouched)
and one MLA layer carrying the retrofit's tensors. The MLA tensors follow the plugin's
config contract:

    q_proj       [heads * head_dim * 2, hidden]   (q | gate)
    kv_a_proj    [latent rank, hidden]            the cached latent
    k_rope_proj  [kv_heads * rope_dim, hidden]    replicated, one block per KV head
    k_up_proj    [heads * nope, rank]             consumed into kv_b_proj at load
    v_up_proj    [heads * head_dim, rank]         consumed into kv_b_proj at load

The Qwen3.5 GDN tensors are exported *unfused* (`in_proj_qkv` / `in_proj_z` /
`in_proj_b` / `in_proj_a`), which is the layout vLLM's weight mapper expects;
transformers exports the fused pair, so this splits it back.
"""

from __future__ import annotations

import json
import pathlib

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

RANK = 64
VOCAB_SIZE = 128


def build(out: pathlib.Path, seed: int = 0, hybrid: bool = True) -> pathlib.Path:
    """Write the checkpoint. ``hybrid=False`` makes every layer an MLA layer.

    The hybrid build is the shape of the released checkpoints (GDN + MLA), but its first
    forward autotunes the stock Flash-Linear-Attention kernels, which takes minutes on
    gfx906 and has nothing to do with this port. The pure-MLA build keeps the fast gate
    fast; the hybrid one is covered by the slow test.
    """
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        layer_types=(
            ["linear_attention", "full_attention"] if hybrid else ["full_attention"] * 2
        ),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        torch_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}

    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    rope_dim = int(head_dim * cfg.rope_parameters["partial_rotary_factor"])
    nope_dim = head_dim - rope_dim

    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    num_v_heads = cfg.linear_num_value_heads
    for layer in range(cfg.num_hidden_layers):
        pre = f"model.layers.{layer}.linear_attn."
        qkvz = state.pop(pre + "in_proj_qkvz.weight", None)
        ba = state.pop(pre + "in_proj_ba.weight", None)
        if qkvz is None or ba is None:
            continue
        split = 2 * key_dim + value_dim
        state[pre + "in_proj_qkv.weight"] = qkvz[:split].contiguous()
        state[pre + "in_proj_z.weight"] = qkvz[split:].contiguous()
        state[pre + "in_proj_b.weight"] = ba[:num_v_heads].contiguous()
        state[pre + "in_proj_a.weight"] = ba[num_v_heads:].contiguous()

    def rnd(*shape):
        return (torch.randn(*shape, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    mla_ranks = (
        {cfg.num_hidden_layers - 1: RANK}
        if hybrid
        else {layer: RANK for layer in range(cfg.num_hidden_layers)}
    )
    for layer, rank in mla_ranks.items():
        pre = f"model.layers.{layer}.self_attn."
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm"):
            state.pop(pre + name + ".weight", None)
        state[pre + "q_proj.weight"] = rnd(heads * head_dim * 2, hidden)
        state[pre + "kv_a_proj.weight"] = rnd(rank, hidden)
        state[pre + "k_rope_proj.weight"] = rnd(kv_heads * rope_dim, hidden)
        state[pre + "k_up_proj.weight"] = rnd(heads * nope_dim, rank)
        state[pre + "v_up_proj.weight"] = rnd(heads * head_dim, rank)
        state[pre + "q_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
        state[pre + "k_norm.weight"] = torch.zeros(head_dim, dtype=torch.bfloat16)

    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / "model.safetensors"))
    _write_tokenizer(out)

    conf = cfg.to_dict()
    conf["architectures"] = ["Qwen3_5MLAForCausalLM"]
    conf["mla_ranks"] = {str(k): v for k, v in mla_ranks.items()}
    conf["mla_latent_groups"] = 1
    # The GDN state page (fixed by the linear-attention shapes) is 8 KiB here; an MLA
    # row of 112 elements gives a 3.5 KiB page that does not divide it, and vLLM 0.32
    # refuses to pad an MLA page. A power-of-two row lands on 4 KiB: block doubles.
    if not hybrid:
        # No linear-attention state page to divide: the scheme is irrelevant, and the
        # default (fewer bytes per row) is what production runs.
        conf.pop("mla_row_scheme", None)
    else:
        conf["mla_row_scheme"] = "pow2"
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    return out


def _write_tokenizer(out: pathlib.Path) -> None:
    """A tiny whitespace tokenizer, so generation needs no tokenizer flag."""
    vocab = {f"w{i}": i for i in range(VOCAB_SIZE - 1)}
    vocab["[UNK]"] = VOCAB_SIZE - 1
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "model_max_length": 512,
            },
            indent=2,
        )
    )
