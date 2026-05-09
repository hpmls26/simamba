"""Correctness, timing, launch-count, and W&B runner for the test kernel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Callable

import torch


def install_lightweight_package_stub() -> None:
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    if "mamba_ssm" not in sys.modules:
        repo_root = Path(__file__).resolve().parents[2]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(repo_root / "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg


install_lightweight_package_stub()

from improved_simamba_kernel import improved_simamba_siso_forward  # noqa: E402
from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (  # noqa: E402
    mamba3_siso_combined as production_triton_simamba,
)
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (  # noqa: E402
    simamba_siso_combined as reference_simamba,
)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def make_inputs(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dtype: torch.dtype,
    device: str,
    include_midpoint: bool,
    include_d: bool,
    include_z: bool,
) -> dict[str, torch.Tensor | None]:
    n_angles = headdim // 2
    value_kwargs = {"device": device, "dtype": dtype}
    coeff_kwargs = {"device": device, "dtype": torch.float32}
    return {
        "Q": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "K": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "V": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "ADT": -0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "DT": 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Simpson": torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Midpoint": torch.rand(batch, nheads, seqlen, **coeff_kwargs) if include_midpoint else None,
        "Q_bias": torch.randn(nheads, headdim, **coeff_kwargs),
        "K_bias": torch.randn(nheads, headdim, **coeff_kwargs),
        "Angles": torch.randn(batch, seqlen, nheads, n_angles, **coeff_kwargs),
        "D": torch.randn(nheads, **coeff_kwargs) if include_d else None,
        "Z": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs) if include_z else None,
    }


def call_reference(inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    return reference_simamba(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Simpson=inputs["Simpson"],
        Midpoint=inputs["Midpoint"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=inputs["D"],
        Z=inputs["Z"],
    )


def call_production_triton(inputs: dict[str, torch.Tensor | None], chunk_size: int) -> torch.Tensor:
    return production_triton_simamba(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Simpson=inputs["Simpson"],
        Midpoint=inputs["Midpoint"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=inputs["D"],
        Z=inputs["Z"],
        chunk_size=chunk_size,
    )


def call_improved(
    inputs: dict[str, torch.Tensor | None],
    chunk_size: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    return improved_simamba_siso_forward(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Simpson=inputs["Simpson"],
        Midpoint=inputs["Midpoint"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=inputs["D"],
        Z=inputs["Z"],
        chunk_size=chunk_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def error_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "max_rel_error": (diff / denom).max().item(),
        "mean_rel_error": (diff / denom).mean().item(),
    }


def cuda_time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
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


def cuda_kernel_summary(fn: Callable[[], torch.Tensor]) -> dict[str, object]:
    from torch.profiler import ProfilerActivity, profile

    def duration_us(event) -> float:
        for attr in ("duration_us", "device_time_total", "self_device_time_total", "cuda_time_total", "self_cuda_time_total"):
            value = getattr(event, attr, None)
            if callable(value):
                value = value()
            if value is not None:
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if value_f > 0.0:
                    return value_f
        time_range = getattr(event, "time_range", None)
        elapsed = getattr(time_range, "elapsed_us", None)
        if callable(elapsed):
            try:
                return float(elapsed())
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()

    rows: dict[str, dict[str, float | int | str]] = {}
    for event in prof.events():
        if "CUDA" not in str(getattr(event, "device_type", "")):
            continue
        row = rows.setdefault(event.name, {"kernel": event.name, "count": 0, "cuda_time_us": 0.0})
        row["count"] = int(row["count"]) + 1
        row["cuda_time_us"] = float(row["cuda_time_us"]) + duration_us(event)

    if not rows:
        for event in prof.key_averages():
            cuda_us = float(getattr(event, "self_cuda_time_total", 0.0))
            if cuda_us <= 0.0:
                continue
            rows[event.key] = {
                "kernel": event.key,
                "count": int(event.count),
                "cuda_time_us": cuda_us,
            }

    sorted_rows = sorted(rows.values(), key=lambda item: float(item["cuda_time_us"]), reverse=True)
    return {
        "cuda_kernel_launches": sum(int(row["count"]) for row in sorted_rows),
        "cuda_kernel_time_us": sum(float(row["cuda_time_us"]) for row in sorted_rows),
        "top_cuda_kernels": sorted_rows[:12],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the fused Simamba test kernel.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--skip-torch-profiler", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-midpoint", action="store_true")
    parser.add_argument("--no-d", action="store_true")
    parser.add_argument("--no-z", action="store_true")
    parser.add_argument("--skip-production", action="store_true")
    parser.add_argument("--only-improved", action="store_true")
    parser.add_argument("--out-dir", default="../results/test_kernel")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="test_kernel_improved_simamba")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the improved Simamba kernel test")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    inputs = make_inputs(
        args.batch,
        args.seqlen,
        args.nheads,
        args.headdim,
        dtype,
        "cuda",
        include_midpoint=not args.no_midpoint,
        include_d=not args.no_d,
        include_z=not args.no_z,
    )

    with torch.no_grad():
        ref = None if args.only_improved else call_reference(inputs)
        prod = None if args.skip_production or args.only_improved else call_production_triton(inputs, args.chunk_size)
        improved = call_improved(inputs, args.chunk_size, args.num_warps, args.num_stages)
        torch.cuda.synchronize()

    empty_error_stats = {
        "max_abs_error": "",
        "mean_abs_error": "",
        "max_rel_error": "",
        "mean_rel_error": "",
    }
    prod_stats = error_stats(prod, ref) if prod is not None and ref is not None else None
    improved_stats = error_stats(improved, ref) if ref is not None else empty_error_stats

    timings = {
        "improved_fused_ms": cuda_time_ms(
            lambda: call_improved(inputs, args.chunk_size, args.num_warps, args.num_stages),
            args.warmup,
            args.iters,
        ),
    }
    if not args.only_improved:
        timings["reference_ms"] = cuda_time_ms(lambda: call_reference(inputs), args.warmup, args.iters)
    if not args.skip_production and not args.only_improved:
        timings["production_triton_ms"] = cuda_time_ms(
            lambda: call_production_triton(inputs, args.chunk_size),
            args.warmup,
            args.iters,
        )
    if args.skip_torch_profiler:
        launch_stats = {
            "improved_fused": {"cuda_kernel_launches": "", "cuda_kernel_time_us": "", "top_cuda_kernels": []},
        }
        if not args.only_improved:
            launch_stats["reference"] = {
                "cuda_kernel_launches": "",
                "cuda_kernel_time_us": "",
                "top_cuda_kernels": [],
            }
        if not args.skip_production and not args.only_improved:
            launch_stats["production_triton"] = {
                "cuda_kernel_launches": "",
                "cuda_kernel_time_us": "",
                "top_cuda_kernels": [],
            }
    else:
        launch_stats = {
            "improved_fused": cuda_kernel_summary(
                lambda: call_improved(inputs, args.chunk_size, args.num_warps, args.num_stages)
            ),
        }
        if not args.only_improved:
            launch_stats["reference"] = cuda_kernel_summary(lambda: call_reference(inputs))
        if not args.skip_production and not args.only_improved:
            launch_stats["production_triton"] = cuda_kernel_summary(
                lambda: call_production_triton(inputs, args.chunk_size)
            )

    base = {
        "timestamp": int(time.time()),
        "batch": args.batch,
        "seqlen": args.seqlen,
        "nheads": args.nheads,
        "headdim": args.headdim,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "midpoint": not args.no_midpoint,
        "d_skip": not args.no_d,
        "z_gate": not args.no_z,
    }
    rows = []
    if prod_stats is not None:
        rows.append({
            **base,
            "kernel": "production_triton",
            "latency_ms": timings["production_triton_ms"],
            "speedup_vs_reference": timings["reference_ms"] / timings["production_triton_ms"],
            "cuda_kernel_launches": launch_stats["production_triton"]["cuda_kernel_launches"],
            **prod_stats,
        })
    rows.extend([
        {
            **base,
            "kernel": "improved_fused",
            "latency_ms": timings["improved_fused_ms"],
            "speedup_vs_reference": (
                timings["reference_ms"] / timings["improved_fused_ms"] if "reference_ms" in timings else ""
            ),
            "cuda_kernel_launches": launch_stats["improved_fused"]["cuda_kernel_launches"],
            **improved_stats,
        },
    ])
    if not args.only_improved:
        rows.append({
            **base,
            "kernel": "reference",
            "latency_ms": timings["reference_ms"],
            "speedup_vs_reference": 1.0,
            "cuda_kernel_launches": launch_stats["reference"]["cuda_kernel_launches"],
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
        })

    root = Path(__file__).resolve().parent
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "improved_kernel_results.csv"
    json_path = out_dir / "improved_kernel_profile.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({"rows": rows, "timings": timings, "launch_stats": launch_stats}, indent=2) + "\n"
    )

    for row in rows:
        print(json.dumps(row), flush=True)
    print(json.dumps({"profile_json": str(json_path), "profile_csv": str(csv_path)}), flush=True)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="test_kernel_improved_simamba",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )
        for row in rows:
            wandb.log(row)
        kernel_tables = {
            "improved_top_cuda_kernels": wandb.Table(columns=["kernel", "count", "cuda_time_us"], data=[
                [item["kernel"], item["count"], item["cuda_time_us"]]
                for item in launch_stats["improved_fused"]["top_cuda_kernels"]
            ])
        }
        if "reference" in launch_stats:
            kernel_tables["reference_top_cuda_kernels"] = wandb.Table(
                columns=["kernel", "count", "cuda_time_us"],
                data=[
                    [item["kernel"], item["count"], item["cuda_time_us"]]
                    for item in launch_stats["reference"]["top_cuda_kernels"]
                ],
            )
        wandb.log(kernel_tables)
        artifact = wandb.Artifact("improved_simamba_test_kernel", type="profile_results")
        artifact.add_file(str(csv_path))
        artifact.add_file(str(json_path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
