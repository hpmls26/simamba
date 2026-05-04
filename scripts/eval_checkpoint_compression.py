#!/usr/bin/env python

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from scripts.train_simamba_lm import FixedValidationSampler, PackedTokenDataset, evaluate, get_amp_dtype


def parse_checkpoint_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoints must be passed as name=/path/to/trainer.pt")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate non-destructive pruning and quantize-dequantize variants of local "
            "training checkpoints on one fixed validation sample set."
        )
    )
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_arg, required=True)
    parser.add_argument("--val-data", type=Path, default=Path("data/slimpajama_500m_50m/val.bin"))
    parser.add_argument("--token-dtype", choices=["uint16", "int32", "int64"], default="uint16")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=20261504)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variants",
        default="baseline,int8_row_qdq,int4_row_qdq,prune_global_10,prune_global_20,prune_global_30",
        help="Comma-separated variants to evaluate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def compression_candidates(model):
    seen = set()
    for name, parameter in model.named_parameters():
        if not torch.is_floating_point(parameter) or parameter.ndim < 2:
            continue
        storage_id = parameter.untyped_storage().data_ptr()
        if storage_id in seen:
            continue
        seen.add(storage_id)
        yield name, parameter


@torch.no_grad()
def quantize_dequantize_rowwise_(model, bits: int):
    qmax = (2 ** (bits - 1)) - 1
    changed = 0
    elements = 0
    for _name, parameter in compression_candidates(model):
        data = parameter.data
        view = data.float().reshape(data.shape[0], -1)
        scale = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        quantized = torch.round(view / scale).clamp(-qmax, qmax)
        data.copy_((quantized * scale).reshape_as(data).to(dtype=data.dtype))
        changed += 1
        elements += data.numel()
    return {"changed_tensors": changed, "candidate_elements": elements, "bits": bits}


@torch.no_grad()
def prune_global_magnitude_(model, fraction: float):
    candidates = list(compression_candidates(model))
    if not candidates:
        return {"changed_tensors": 0, "candidate_elements": 0, "sparsity": 0.0}
    values = torch.cat([parameter.detach().abs().float().flatten().cpu() for _name, parameter in candidates])
    threshold = torch.quantile(values, float(fraction)).to(next(model.parameters()).device)
    changed = 0
    total = 0
    zeros = 0
    for _name, parameter in candidates:
        mask = parameter.detach().abs() <= threshold
        parameter.data.masked_fill_(mask, 0)
        changed += 1
        total += parameter.numel()
        zeros += int((parameter.detach() == 0).sum().item())
    return {
        "changed_tensors": changed,
        "candidate_elements": total,
        "threshold": float(threshold.detach().cpu()),
        "sparsity": zeros / max(1, total),
    }


def apply_variant(model, variant: str):
    if variant == "baseline":
        return {"changed_tensors": 0, "candidate_elements": sum(p.numel() for _n, p in compression_candidates(model))}
    if variant == "int8_row_qdq":
        return quantize_dequantize_rowwise_(model, bits=8)
    if variant == "int4_row_qdq":
        return quantize_dequantize_rowwise_(model, bits=4)
    if variant.startswith("prune_global_"):
        percent = float(variant.rsplit("_", 1)[1])
        return prune_global_magnitude_(model, fraction=percent / 100.0)
    raise ValueError(f"Unknown variant: {variant}")


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MambaConfig(**ckpt["config"])
    model = MambaLMHeadModel(config=config, device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output = Path("run_logs") / f"compression_eval_{stamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".md")

    val_data = PackedTokenDataset(args.val_data, args.token_dtype)
    eval_args = SimpleNamespace(
        eval_iters=args.eval_iters,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
    )
    sampler = FixedValidationSampler(
        val_data,
        batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        device=device,
        eval_iters=args.eval_iters,
        rank=0,
        world_size=1,
        seed=args.eval_seed,
    )
    amp_dtype = get_amp_dtype(args.dtype)
    amp_ctx = (
        (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype))
        if args.dtype != "fp32"
        else nullcontext
    )
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    results = []

    with output.open("w") as handle:
        for checkpoint_name, checkpoint_path in args.checkpoint:
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            for variant in variants:
                start = time.perf_counter()
                model, ckpt = load_model(checkpoint_path, device)
                model.eval()
                compression = apply_variant(model, variant)
                loss = float(evaluate(model, val_data, eval_args, device, amp_ctx, sampler).item())
                result = {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path),
                    "variant": variant,
                    "loss": loss,
                    "ppl": math.exp(loss) if loss < 20 else None,
                    "duration_sec": round(time.perf_counter() - start, 3),
                    "step": ckpt.get("completed_step", ckpt.get("step")),
                    "config": ckpt.get("config", {}),
                    "compression": compression,
                    "eval": {
                        "val_data": str(args.val_data),
                        "eval_iters": args.eval_iters,
                        "micro_batch_size": args.micro_batch_size,
                        "seq_len": args.seq_len,
                        "eval_seed": args.eval_seed,
                        "dtype": args.dtype,
                    },
                }
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint_name,
                            "variant": variant,
                            "loss": loss,
                            "duration_sec": result["duration_sec"],
                        }
                    ),
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    by_checkpoint = {}
    for result in results:
        by_checkpoint.setdefault(result["checkpoint"], []).append(result)

    lines = [
        "# Compression Evaluation",
        "",
        f"- Validation data: `{args.val_data}`",
        f"- Fixed eval batches: `{args.eval_iters}` x `{args.micro_batch_size}` x `{args.seq_len}` tokens",
        f"- Eval seed: `{args.eval_seed}`",
        "",
        "| Checkpoint | Variant | Val loss | Delta vs baseline | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for checkpoint, checkpoint_results in by_checkpoint.items():
        baseline = next(item for item in checkpoint_results if item["variant"] == "baseline")
        baseline_loss = baseline["loss"]
        for item in checkpoint_results:
            compression = item["compression"]
            note = ""
            if "bits" in compression:
                note = f"{compression['bits']}-bit row qdq"
            elif "sparsity" in compression:
                note = f"{compression['sparsity']:.1%} candidate sparsity"
            lines.append(
                "| {checkpoint} | `{variant}` | {loss:.6f} | {delta:+.6f} | {note} |".format(
                    checkpoint=checkpoint,
                    variant=item["variant"],
                    loss=item["loss"],
                    delta=item["loss"] - baseline_loss,
                    note=note,
                )
            )
    lines.append("")
    lines.append(
        "Quantization here is quantize-dequantize weight perturbation for validation loss, "
        "not an optimized int kernel benchmark."
    )
    summary_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"jsonl": str(output), "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
