#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Stream and tokenize a bounded SlimPajama sample.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--dataset-name", default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--eos-after-each-doc", action="store_true", default=True)
    parser.add_argument("--no-eos-after-each-doc", dest="eos_after_each_doc", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer} has no eos_token_id.")

    total_tokens = args.train_tokens + args.val_tokens
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    all_tokens = np.memmap(
        args.output_dir / "all_tokens.uint16.bin",
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    cursor = 0
    docs = 0
    for example in dataset:
        text = example.get("text")
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if args.eos_after_each_doc:
            token_ids.append(eos_id)
        if not token_ids:
            continue
        remaining = total_tokens - cursor
        if remaining <= 0:
            break
        chunk = np.asarray(token_ids[:remaining], dtype=np.uint16)
        all_tokens[cursor : cursor + len(chunk)] = chunk
        cursor += len(chunk)
        docs += 1
        if docs % 1000 == 0:
            print(f"docs={docs} tokens={cursor}/{total_tokens}", flush=True)

    if cursor < total_tokens:
        raise RuntimeError(f"Only collected {cursor} tokens, expected {total_tokens}.")

    all_tokens.flush()
    del all_tokens

    data = np.memmap(args.output_dir / "all_tokens.uint16.bin", mode="r", dtype=np.uint16, shape=(total_tokens,))
    train = np.memmap(args.output_dir / "train.bin", mode="w+", dtype=np.uint16, shape=(args.train_tokens,))
    val = np.memmap(args.output_dir / "val.bin", mode="w+", dtype=np.uint16, shape=(args.val_tokens,))
    train[:] = data[: args.train_tokens]
    val[:] = data[args.train_tokens :]
    train.flush()
    val.flush()
    del train
    del val
    del data
    (args.output_dir / "meta.txt").write_text(
        "\n".join(
            [
                f"dataset={args.dataset_name}",
                f"split={args.dataset_split}",
                f"tokenizer={args.tokenizer}",
                f"train_tokens={args.train_tokens}",
                f"val_tokens={args.val_tokens}",
                f"seed={args.seed}",
            ]
        )
        + "\n"
    )
    print(f"Wrote {args.output_dir / 'train.bin'} and {args.output_dir / 'val.bin'}", flush=True)


if __name__ == "__main__":
    main()
