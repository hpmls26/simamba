import argparse
import csv
import os
import sys
import time
import types
from pathlib import Path

import torch
import wandb


def install_lightweight_package_stub():
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    if "mamba_ssm" not in sys.modules:
        repo_root = Path(__file__).resolve().parents[1]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(repo_root / "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg


install_lightweight_package_stub()

from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (  # noqa: E402
    mamba3_siso_combined as simamba_triton_siso_combined,
)
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (  # noqa: E402
    simamba_siso_combined as simamba_reference_siso_combined,
)


INPUT_NAMES = ("Q", "K", "V", "ADT", "DT", "Simpson", "Midpoint", "Q_bias", "K_bias", "Angles", "D", "Z")


def dtype_from_name(name):
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def clone_for_grad(inputs):
    cloned = {}
    for name, tensor in inputs.items():
        cloned[name] = tensor.detach().clone().requires_grad_(True)
    return cloned


def make_inputs(batch, seqlen, nheads, headdim_qk, headdim_v, dtype, device):
    n_angles = headdim_qk // 2
    value_kwargs = {"device": device, "dtype": dtype}
    coeff_kwargs = {"device": device, "dtype": torch.float32}
    return {
        "Q": torch.randn(batch, seqlen, nheads, headdim_qk, **value_kwargs),
        "K": torch.randn(batch, seqlen, nheads, headdim_qk, **value_kwargs),
        "V": torch.randn(batch, seqlen, nheads, headdim_v, **value_kwargs),
        "ADT": -0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "DT": 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Simpson": torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Midpoint": torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Q_bias": torch.randn(nheads, headdim_qk, **coeff_kwargs),
        "K_bias": torch.randn(nheads, headdim_qk, **coeff_kwargs),
        "Angles": torch.randn(batch, seqlen, nheads, n_angles, **coeff_kwargs),
        "D": torch.randn(nheads, **coeff_kwargs),
        "Z": torch.randn(batch, seqlen, nheads, headdim_v, **value_kwargs),
    }


def call_kernel(fn, inputs):
    return fn(
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
        return_final_states=False,
    )


def error_stats(got, ref):
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "max_rel_error": (diff / denom).max().item(),
        "mean_rel_error": (diff / denom).mean().item(),
    }


def status_from_error(stats, atol, rtol):
    return "pass" if stats["max_abs_error"] <= atol or stats["max_rel_error"] <= rtol else "fail"


def run_case(args):
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    inputs = make_inputs(args.batch, args.seqlen, args.nheads, args.headdim, args.headdim, dtype, "cuda")
    rows = []

    with torch.no_grad():
        ref_out = call_kernel(simamba_reference_siso_combined, inputs)
        tri_out = call_kernel(simamba_triton_siso_combined, inputs)
    stats = error_stats(tri_out, ref_out)
    rows.append({
        "timestamp": int(time.time()),
        "check": "forward",
        "tensor": "output",
        "batch": args.batch,
        "seqlen": args.seqlen,
        "nheads": args.nheads,
        "headdim": args.headdim,
        "dtype": args.dtype,
        "status": status_from_error(stats, args.forward_atol, args.forward_rtol),
        **stats,
    })

    ref_inputs = clone_for_grad(inputs)
    tri_inputs = clone_for_grad(inputs)
    grad_out = torch.randn_like(ref_out)
    ref_train_out = call_kernel(simamba_reference_siso_combined, ref_inputs)
    tri_train_out = call_kernel(simamba_triton_siso_combined, tri_inputs)
    torch.autograd.backward(ref_train_out, grad_tensors=grad_out)
    torch.autograd.backward(tri_train_out, grad_tensors=grad_out)

    for name in INPUT_NAMES:
        stats = error_stats(tri_inputs[name].grad, ref_inputs[name].grad)
        rows.append({
            "timestamp": int(time.time()),
            "check": "backward",
            "tensor": name,
            "batch": args.batch,
            "seqlen": args.seqlen,
            "nheads": args.nheads,
            "headdim": args.headdim,
            "dtype": args.dtype,
            "status": status_from_error(stats, args.backward_atol, args.backward_rtol),
            **stats,
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Check | Tensor | Status | Max abs err | Mean abs err | Max rel err | Mean rel err |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['check']} | {row['tensor']} | {row['status']} | "
            f"{row['max_abs_error']:.4e} | {row['mean_abs_error']:.4e} | "
            f"{row['max_rel_error']:.4e} | {row['mean_rel_error']:.4e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build Simamba Triton-vs-reference correctness tables.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=8)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--forward-atol", type=float, default=5e-2)
    parser.add_argument("--forward-rtol", type=float, default=5e-2)
    parser.add_argument("--backward-atol", type=float, default=1e-2)
    parser.add_argument("--backward-rtol", type=float, default=1e-2)
    parser.add_argument("--out", default="results/kernel_correctness_reference.csv")
    parser.add_argument("--md-out", default="results/kernel_correctness_reference.md")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="kernel_correctness")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton correctness checks")

    root = Path(__file__).resolve().parent
    rows = run_case(args)
    out_path = root / args.out
    md_path = root / args.md_out
    write_csv(out_path, rows)
    write_markdown(md_path, rows)
    for row in rows:
        print(row, flush=True)
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="kernel_correctness",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )
        for row in rows:
            wandb.log(row)
        artifact = wandb.Artifact("kernel_correctness_reference", type="profile_results")
        artifact.add_file(str(out_path))
        artifact.add_file(str(md_path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
