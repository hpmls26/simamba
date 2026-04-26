#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="Stream and tokenize a bounded SlimPajama sample.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--test-tokens", type=int, default=0)
    parser.add_argument("--dataset-name", default="MBZUAI-LLM/SlimPajama-627B-DC")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--eos-after-each-doc", action="store_true", default=True)
    parser.add_argument("--no-eos-after-each-doc", dest="eos_after_each_doc", action="store_false")
    parser.add_argument(
        "--allowed-sources",
        nargs="*",
        default=None,
        help=(
            "Optional allowed SlimPajama source names, e.g. RedPajamaArXiv RedPajamaWikipedia. "
            "Matches meta.redpajama_set_name."
        ),
    )
    return parser.parse_args()


def get_source_name(example) -> Optional[str]:
    meta = example.get("meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("redpajama_set_name")


def iter_filtered(dataset: Iterable, allowed_sources: Optional[set]):
    if not allowed_sources:
        yield from dataset
        return
    for example in dataset:
        source = get_source_name(example)
        if source in allowed_sources:
            yield example


def open_stream(dataset_name: str, dataset_config: str, split: str, seed: int, shuffle_buffer: int):
    from datasets import load_dataset

    try:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to open SlimPajama from Hugging Face. "
            f"dataset={dataset_name!r} config={dataset_config!r} split={split!r}. "
            "Try:\n"
            "  .venv/bin/pip install -U datasets huggingface_hub hf-xet\n"
            f"Original error: {exc}"
        ) from exc
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def write_split(
    *,
    split_name: str,
    token_target: int,
    dataset_name: str,
    dataset_config: str,
    output_path: Path,
    tokenizer,
    eos_after_each_doc: bool,
    allowed_sources: Optional[set],
    seed: int,
    shuffle_buffer: int,
):
    if token_target <= 0:
        return {"split": split_name, "tokens": 0, "docs": 0, "path": str(output_path), "sources": sorted(allowed_sources) if allowed_sources else None}

    print(f"[prepare] Opening split={split_name} target_tokens={token_target}", flush=True)
    dataset = open_stream(dataset_name, dataset_config, split_name, seed, shuffle_buffer)
    dataset = iter_filtered(dataset, allowed_sources)

    mmap = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_target,))
    eos_id = tokenizer.eos_token_id
    cursor = 0
    docs = 0
    start = time.time()
    pbar = tqdm(total=token_target, unit="tok", desc=f"{split_name}", dynamic_ncols=True) if tqdm is not None else None

    try:
        for example in dataset:
            text = example.get("text")
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if eos_after_each_doc:
                token_ids.append(eos_id)
            if not token_ids:
                continue

            remaining = token_target - cursor
            if remaining <= 0:
                break

            chunk = np.asarray(token_ids[:remaining], dtype=np.uint16)
            mmap[cursor : cursor + len(chunk)] = chunk
            cursor += len(chunk)
            docs += 1

            if pbar is not None:
                pbar.update(len(chunk))
            elif docs % 100 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"[prepare] split={split_name} docs={docs} tokens={cursor}/{token_target} "
                    f"({cursor / token_target:.1%}) toks_per_sec={cursor / elapsed:.0f}",
                    flush=True,
                )
    except ValueError as exc:
        if "Compression type zstd not supported" in str(exc):
            raise RuntimeError(
                "Your environment is missing zstd support required to stream this dataset. "
                "Install it in the repo venv with:\n"
                "  .venv/bin/pip install zstandard\n"
                "and rerun the prep job."
            ) from exc
        raise

    if pbar is not None:
        pbar.close()

    if cursor < token_target:
        mmap.flush()
        del mmap
        raise RuntimeError(
            f"Split {split_name!r} only produced {cursor} tokens, expected {token_target}. "
            "Lower the target token count or widen the source filter."
        )

    mmap.flush()
    del mmap
    elapsed = max(time.time() - start, 1e-6)
    print(
        f"[prepare] Completed split={split_name} docs={docs} tokens={cursor} "
        f"elapsed_s={elapsed:.1f} toks_per_sec={cursor / elapsed:.0f} path={output_path}",
        flush=True,
    )
    return {
        "split": split_name,
        "tokens": cursor,
        "docs": docs,
        "path": str(output_path),
        "sources": sorted(allowed_sources) if allowed_sources else None,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[prepare] Starting SlimPajama sample preparation", flush=True)
    print(f"[prepare] dataset={args.dataset_name} config={args.dataset_config}", flush=True)
    print(f"[prepare] output_dir={args.output_dir}", flush=True)
    print(f"[prepare] tokenizer={args.tokenizer}", flush=True)
    print(
        f"[prepare] targets train={args.train_tokens} val={args.val_tokens} test={args.test_tokens}",
        flush=True,
    )
    print(f"[prepare] shuffle_buffer={args.shuffle_buffer} seed={args.seed}", flush=True)
    if args.allowed_sources:
        print(f"[prepare] allowed_sources={args.allowed_sources}", flush=True)
    else:
        print("[prepare] allowed_sources=<all>", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer} has no eos_token_id.")

    allowed_sources = set(args.allowed_sources) if args.allowed_sources else None
    results = []
    results.append(
        write_split(
            split_name="train",
            token_target=args.train_tokens,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            output_path=args.output_dir / "train.bin",
            tokenizer=tokenizer,
            eos_after_each_doc=args.eos_after_each_doc,
            allowed_sources=allowed_sources,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    )
    results.append(
        write_split(
            split_name="validation",
            token_target=args.val_tokens,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            output_path=args.output_dir / "val.bin",
            tokenizer=tokenizer,
            eos_after_each_doc=args.eos_after_each_doc,
            allowed_sources=allowed_sources,
            seed=args.seed + 1,
            shuffle_buffer=args.shuffle_buffer,
        )
    )
    if args.test_tokens > 0:
        results.append(
            write_split(
                split_name="test",
                token_target=args.test_tokens,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                output_path=args.output_dir / "test.bin",
                tokenizer=tokenizer,
                eos_after_each_doc=args.eos_after_each_doc,
                allowed_sources=allowed_sources,
                seed=args.seed + 2,
                shuffle_buffer=args.shuffle_buffer,
            )
        )

    meta = {
        "dataset": args.dataset_name,
        "config": args.dataset_config,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "eos_after_each_doc": args.eos_after_each_doc,
        "allowed_sources": args.allowed_sources,
        "outputs": results,
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[prepare] wrote {meta_path}", flush=True)
    print("[prepare] Done", flush=True)


if __name__ == "__main__":
    main()
