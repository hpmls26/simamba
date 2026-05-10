#!/usr/bin/env python
"""Capture one warmed vLLM generation for Nsight decode analysis."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import vllm_sweep


def make_prompt(words: int) -> str:
    base = (
        "The quick brown fox studies sequence models, numerical integration, "
        "and GPU kernels for language modeling throughput."
    ).split()
    return " ".join(base[i % len(base)] for i in range(words))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--simamba-vllm-fork", required=True)
    parser.add_argument("--simamba-native-backend", default="triton")
    parser.add_argument("--prompt-words", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--mamba-block-size", type=int, default=16)
    parser.add_argument("--use-cudagraph", action="store_true")
    args = parser.parse_args()

    vllm_sweep.enable_simamba_vllm_fork(args.simamba_vllm_fork)
    vllm_sweep.SIMAMBA_NATIVE_BACKEND = args.simamba_native_backend

    cache_kwargs = {}
    if args.prefix_cache:
        cache_kwargs["mamba_block_size"] = args.mamba_block_size

    llm = LLM(
        model=args.model,
        tokenizer="EleutherAI/gpt-neox-20b",
        trust_remote_code=True,
        model_impl="auto",
        enforce_eager=not args.use_cudagraph,
        max_model_len=1024,
        gpu_memory_utilization=0.25,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        hf_overrides=vllm_sweep.profiling_hf_overrides,
        enable_prefix_caching=args.prefix_cache,
        disable_log_stats=False,
        **cache_kwargs,
    )

    prompt = make_prompt(args.prompt_words)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    for _ in range(args.warmup):
        llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.cuda.cudart().cudaProfilerStop()

    generated = len(outputs[0].outputs[0].token_ids)
    print(
        {
            "elapsed_sec": elapsed,
            "generated_tokens": generated,
            "tokens_per_sec": generated / elapsed if elapsed else 0.0,
            "simamba_native_backend": args.simamba_native_backend,
            "prefix_cache": args.prefix_cache,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
