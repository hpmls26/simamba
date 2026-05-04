#!/usr/bin/env python3
"""Convert our training Mamba2 checkpoint to standard HF/vLLM layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import Mamba2Config


DEFAULT_SOURCE = Path("outputs/disc10m_mamba2_fp32_500m_20260502_185217/best")
DEFAULT_OUTPUT = Path("hf_exports/mamba2-10m-slimpajama-500m-vllm")
TOKENIZER_SOURCE = Path("hf_exports/mamba2-10m-slimpajama-500m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_hf_config(training_config: dict) -> Mamba2Config:
    ssm_cfg = training_config["ssm_cfg"]
    if ssm_cfg.get("layer") != "Mamba2":
        raise ValueError(f"Expected a Mamba2 checkpoint, got {ssm_cfg.get('layer')!r}.")

    hidden_size = int(training_config["d_model"])
    expand = int(ssm_cfg["expand"])
    head_dim = int(ssm_cfg["headdim"])
    inner_size = hidden_size * expand
    if inner_size % head_dim != 0:
        raise ValueError(
            f"hidden_size * expand must be divisible by head_dim, got {inner_size} and {head_dim}."
        )

    cfg = Mamba2Config(
        vocab_size=int(training_config["vocab_size"]),
        hidden_size=hidden_size,
        state_size=int(ssm_cfg["d_state"]),
        num_hidden_layers=int(training_config["n_layer"]),
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=0,
        expand=expand,
        conv_kernel=int(ssm_cfg["d_conv"]),
        n_groups=int(ssm_cfg.get("ngroups", 1)),
        head_dim=head_dim,
        num_heads=inner_size // head_dim,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        residual_in_fp32=bool(training_config.get("residual_in_fp32", True)),
        rms_norm=bool(training_config.get("rms_norm", True)),
        chunk_size=int(ssm_cfg.get("chunk_size", 16)),
        tie_word_embeddings=bool(training_config.get("tie_embeddings", True)),
        torch_dtype="float32",
    )
    cfg.architectures = ["Mamba2ForCausalLM"]
    return cfg


def convert_state_dict(source: Path, output: Path) -> None:
    state_dict = torch.load(source / "pytorch_model.bin", map_location="cpu")
    converted = {}
    for name, tensor in state_dict.items():
        if name == "backbone.embedding.weight":
            name = "backbone.embeddings.weight"
        converted[name] = tensor
    torch.save(converted, output / "pytorch_model.bin")


def write_vllm_config(config: Mamba2Config, output: Path) -> None:
    data = config.to_dict()
    data["architectures"] = ["Mamba2ForCausalLM"]
    data["intermediate_size"] = int(data["hidden_size"]) * int(data["expand"])
    data["norm_before_gate"] = False
    data["torch_dtype"] = "float32"
    data["dtype"] = "float32"
    data["time_step_limit"] = [0.0, float("inf")]
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    training_config = json.loads((source / "config.json").read_text())
    hf_config = build_hf_config(training_config)
    write_vllm_config(hf_config, output)

    convert_state_dict(source, output)

    for filename in ("metrics.json", "checkpoint_manifest.json"):
        shutil.copy2(source / filename, output / filename)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(TOKENIZER_SOURCE / filename, output / filename)

    print(output)


if __name__ == "__main__":
    main()
