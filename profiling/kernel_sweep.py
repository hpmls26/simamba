import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import wandb


KERNELS = {
    "simamba": ("simamba", "simamba_siso_combined", "simamba_siso_combined"),
    "mamba3": ("mamba3", "mamba3_siso_combined", "mamba3_siso_combined"),
}


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


@contextmanager
def import_path(path):
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


def load_kernel(root, kernel_name):
    kernel_dir, module_name, fn_name = KERNELS[kernel_name]
    with import_path(root / kernel_dir):
        module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def make_inputs(batch, seqlen, nheads, headdim, dtype, device, requires_grad=False):
    n_angles = headdim // 2
    tensor_kwargs = {"device": device, "dtype": dtype}
    q = torch.randn(batch, seqlen, nheads, headdim, **tensor_kwargs)
    k = torch.randn(batch, seqlen, nheads, headdim, **tensor_kwargs)
    v = torch.randn(batch, seqlen, nheads, headdim, **tensor_kwargs)
    adt = torch.randn(batch, nheads, seqlen, **tensor_kwargs)
    dt = torch.randn(batch, nheads, seqlen, **tensor_kwargs)
    simpson = torch.randn(batch, nheads, seqlen, **tensor_kwargs)
    q_bias = torch.randn(nheads, headdim, **tensor_kwargs)
    k_bias = torch.randn(nheads, headdim, **tensor_kwargs)
    angles = torch.randn(batch, seqlen, nheads, n_angles, **tensor_kwargs)
    tensors = [q, k, v, adt, dt, simpson, q_bias, k_bias, angles]
    if requires_grad:
        for tensor in tensors:
            tensor.requires_grad_(True)
    return tensors


def cuda_time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def quantiles(values):
    values = sorted(values)
    if not values:
        return math.nan, math.nan
    p50 = values[len(values) // 2]
    p95 = values[min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)]
    return p50, p95


def benchmark_forward(fn, inputs, warmup, iters):
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        samples.append(cuda_time_ms(lambda: fn(*inputs), warmup=0, iters=1))
    peak_mem = torch.cuda.max_memory_allocated()
    p50, p95 = quantiles(samples)
    return {
        "forward_ms_mean": sum(samples) / len(samples),
        "forward_ms_p50": p50,
        "forward_ms_p95": p95,
        "peak_memory_bytes": peak_mem,
    }


def benchmark_backward(fn, inputs, warmup, iters):
    def step():
        for tensor in inputs:
            tensor.grad = None
        output = fn(*inputs)
        loss = output.float().sum()
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    try:
        elapsed = cuda_time_ms(step, warmup=warmup, iters=iters)
    except Exception as exc:
        torch.cuda.empty_cache()
        return {"backward_status": f"failed: {type(exc).__name__}: {exc}"}
    return {
        "backward_status": "ok",
        "forward_backward_ms_mean": elapsed,
        "peak_train_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def compare_outputs(functions, inputs, baseline):
    with torch.no_grad():
        base = functions[baseline](*inputs).float()
        rows = []
        for name, fn in functions.items():
            if name == baseline:
                continue
            try:
                out = fn(*inputs).float()
            except Exception as exc:
                rows.append({
                    "baseline": baseline,
                    "candidate": name,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                    "max_abs_error": "",
                    "mean_abs_error": "",
                    "max_rel_error": "",
                    "mean_rel_error": "",
                })
                continue
            diff = (out - base).abs()
            denom = base.abs().clamp_min(1e-6)
            rows.append({
                "baseline": baseline,
                "candidate": name,
                "status": "ok",
                "max_abs_error": diff.max().item(),
                "mean_abs_error": diff.mean().item(),
                "max_rel_error": (diff / denom).max().item(),
                "mean_rel_error": (diff / denom).mean().item(),
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep Simamba/Euler kernel latency and memory.")
    parser.add_argument("--kernels", default="simamba,mamba3")
    parser.add_argument("--seq-lens", default="128,256,512,1024")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--out", default="results/kernel_sweep.csv")
    parser.add_argument("--correctness-out", default="results/kernel_correctness.csv")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="kernel_sweep")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel profiling")

    root = Path(__file__).resolve().parent
    kernels = [name for name in args.kernels.split(",") if name]
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    functions = {name: load_kernel(root, name) for name in kernels}
    run = None
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="kernel_sweep",
            group=args.wandb_group,
            config=vars(args),
        )

    metric_rows = []
    correctness_rows = []
    for batch in parse_ints(args.batch_sizes):
        for seqlen in parse_ints(args.seq_lens):
            base_inputs = make_inputs(batch, seqlen, args.nheads, args.headdim, dtype, "cuda")
            for name, fn in functions.items():
                inputs = [tensor.detach().clone() for tensor in base_inputs]
                row = {
                    "timestamp": int(time.time()),
                    "kernel": name,
                    "batch": batch,
                    "seqlen": seqlen,
                    "nheads": args.nheads,
                    "headdim": args.headdim,
                    "dtype": args.dtype,
                }
                try:
                    row.update(benchmark_forward(fn, inputs, args.warmup, args.iters))
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = f"failed: {type(exc).__name__}: {exc}"
                if args.include_backward and row["status"] == "ok":
                    train_inputs = make_inputs(
                        batch, seqlen, args.nheads, args.headdim, dtype, "cuda", requires_grad=True
                    )
                    row.update(benchmark_backward(fn, train_inputs, args.warmup, max(1, args.iters // 4)))
                metric_rows.append(row)
                print(json.dumps(row), flush=True)
                if run is not None:
                    wandb.log(row)
            if args.compare and len(functions) > 1:
                for row in compare_outputs(functions, base_inputs, baseline=kernels[0]):
                    row.update({
                        "batch": batch,
                        "seqlen": seqlen,
                        "nheads": args.nheads,
                        "headdim": args.headdim,
                        "dtype": args.dtype,
                    })
                    correctness_rows.append(row)

    write_csv(root / args.out, metric_rows)
    write_csv(root / args.correctness_out, correctness_rows)
    if run is not None:
        artifact = wandb.Artifact("kernel_sweep_results", type="profile_results")
        artifact.add_file(str(root / args.out))
        if correctness_rows:
            artifact.add_file(str(root / args.correctness_out))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
