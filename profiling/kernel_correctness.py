import argparse
import csv
import math
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb


FIELDS = (
    "timestamp",
    "kernel",
    "reference",
    "check",
    "tensor",
    "batch",
    "seqlen",
    "nheads",
    "headdim",
    "dtype",
    "status",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "notes",
)

SIMAMBA_INPUT_NAMES = (
    "Q",
    "K",
    "V",
    "ADT",
    "DT",
    "Simpson",
    "Midpoint",
    "Q_bias",
    "K_bias",
    "Angles",
    "D",
    "Z",
)
MAMBA3_INPUT_NAMES = ("Q", "K", "V", "ADT", "DT", "Simpson", "Q_bias", "K_bias", "Angles")


def install_lightweight_package_stub():
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    if "mamba_ssm" not in sys.modules:
        repo_root = Path(__file__).resolve().parents[1]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(repo_root / "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg


install_lightweight_package_stub()

ROOT = Path(__file__).resolve().parent
TEST_KERNEL_DIR = ROOT / "test_kernel"
if str(TEST_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_KERNEL_DIR))

from improved_simamba_kernel import improved_simamba_siso_forward  # noqa: E402
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (  # noqa: E402
    mamba3_siso_combined as mamba3_triton_siso_combined,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (  # noqa: E402
    mamba3_siso_combined as simamba_triton_siso_combined,
)
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (  # noqa: E402
    simamba_siso_combined as simamba_reference_siso_combined,
)


def dtype_from_name(name):
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def parse_kernel_list(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


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


def call_simamba_reference(inputs, args):
    return simamba_reference_siso_combined(
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
        chunk_size=args.chunk_size,
        return_final_states=False,
    )


def call_simamba_triton(inputs, args):
    return simamba_triton_siso_combined(
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
        chunk_size=args.chunk_size,
        return_final_states=False,
    )


def apply_rotary(tensor, cos, sin):
    tensor_pair = tensor.reshape(*tensor.shape[:-1], -1, 2)
    tensor_0 = tensor_pair[..., 0]
    tensor_1 = tensor_pair[..., 1]
    if cos.shape[-1] < tensor_0.shape[-1]:
        pad_size = tensor_0.shape[-1] - cos.shape[-1]
        cos = F.pad(cos, (0, pad_size), value=1.0)
        sin = F.pad(sin, (0, pad_size), value=0.0)
    rotated_0 = tensor_0 * cos - tensor_1 * sin
    rotated_1 = tensor_0 * sin + tensor_1 * cos
    return torch.stack((rotated_0, rotated_1), dim=-1).reshape_as(tensor)


def mamba3_reference_siso_combined(
    Q,
    K,
    V,
    ADT,
    DT,
    Trap,
    Q_bias,
    K_bias,
    Angles,
    D=None,
    Z=None,
):
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    n_angles = Angles.shape[-1]
    if nheads_qk != nheads:
        gqa = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa, dim=2)
        K = K.repeat_interleave(gqa, dim=2)

    angle_state = torch.zeros((batch, nheads, n_angles), dtype=torch.float32, device=Q.device)
    ssm_state = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=Q.device)
    k_state = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=Q.device)
    v_state = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=V.device)
    angles = torch.tanh(Angles.float()) * math.pi
    outputs = []

    for idx in range(seqlen):
        q = Q[:, idx] + Q_bias.unsqueeze(0).to(Q.dtype)
        k = K[:, idx] + K_bias.unsqueeze(0).to(K.dtype)
        v = V[:, idx]
        adt = ADT[:, :, idx].float()
        dt = DT[:, :, idx].float()
        trap = torch.sigmoid(Trap[:, :, idx].float())

        angle_state = torch.remainder(angle_state + angles[:, idx] * dt.unsqueeze(-1), 2.0 * math.pi)
        q_rot = apply_rotary(q, torch.cos(angle_state), torch.sin(angle_state))
        k_rot = apply_rotary(k, torch.cos(angle_state), torch.sin(angle_state))

        alpha = torch.exp(adt)
        beta = (1.0 - trap) * dt * alpha
        gamma = trap * dt
        ssm_state = alpha[:, :, None, None] * ssm_state
        ssm_state = ssm_state + beta[:, :, None, None] * (
            v_state.float().unsqueeze(-1) * k_state.float().unsqueeze(-2)
        )
        ssm_state = ssm_state + gamma[:, :, None, None] * (
            v.float().unsqueeze(-1) * k_rot.float().unsqueeze(-2)
        )

        out = torch.einsum("bhvd,bhd->bhv", ssm_state, q_rot.float()).to(V.dtype)
        if D is not None:
            out = out + D[None, :, None].to(out.dtype) * v
        if Z is not None:
            z = Z[:, idx]
            out = out * z * torch.sigmoid(z)
        outputs.append(out)
        k_state = k_rot
        v_state = v

    return torch.stack(outputs, dim=1)


def call_mamba3_reference(inputs, args):
    return mamba3_reference_siso_combined(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Trap=inputs["Simpson"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=None,
        Z=None,
    )


def call_mamba3_triton(inputs, args):
    return mamba3_triton_siso_combined(
        Q=inputs["Q"],
        K=inputs["K"],
        V=inputs["V"],
        ADT=inputs["ADT"],
        DT=inputs["DT"],
        Trap=inputs["Simpson"],
        Q_bias=inputs["Q_bias"],
        K_bias=inputs["K_bias"],
        Angles=inputs["Angles"],
        D=None,
        Z=None,
        chunk_size=args.chunk_size,
        return_final_states=False,
    )


def call_improved(inputs, args):
    return improved_simamba_siso_forward(
        inputs["Q"],
        inputs["K"],
        inputs["V"],
        inputs["ADT"],
        inputs["DT"],
        inputs["Simpson"],
        inputs["Q_bias"],
        inputs["K_bias"],
        inputs["Angles"],
        Midpoint=inputs["Midpoint"],
        D=inputs["D"],
        Z=inputs["Z"],
        chunk_size=args.chunk_size,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


CASE_SPECS = {
    "mamba3": {
        "tested": call_mamba3_triton,
        "reference": call_mamba3_reference,
        "reference_name": "pytorch_mamba3_reference",
        "backward_inputs": MAMBA3_INPUT_NAMES,
        "supports_backward": True,
        "notes": "D/Z disabled for core Mamba3 check",
    },
    "simamba": {
        "tested": call_simamba_triton,
        "reference": call_simamba_reference,
        "reference_name": "pytorch_simamba_reference",
        "backward_inputs": SIMAMBA_INPUT_NAMES,
        "supports_backward": True,
    },
    "improved": {
        "tested": call_improved,
        "reference": call_simamba_reference,
        "reference_name": "pytorch_simamba_reference",
        "backward_inputs": (),
        "supports_backward": False,
    },
}


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


def base_row(args, kernel, reference, check, tensor, status, notes=""):
    return {
        "timestamp": int(time.time()),
        "kernel": kernel,
        "reference": reference,
        "check": check,
        "tensor": tensor,
        "batch": args.batch,
        "seqlen": args.seqlen,
        "nheads": args.nheads,
        "headdim": args.headdim,
        "dtype": args.dtype,
        "status": status,
        "max_abs_error": "",
        "mean_abs_error": "",
        "max_rel_error": "",
        "mean_rel_error": "",
        "notes": notes,
    }


def result_row(args, kernel, reference, check, tensor, status, stats, notes=""):
    row = base_row(args, kernel, reference, check, tensor, status, notes)
    row.update(stats)
    return row


def run_forward(kernel, spec, inputs, args):
    with torch.no_grad():
        ref_out = spec["reference"](inputs, args)
        tested_out = spec["tested"](inputs, args)
    stats = error_stats(tested_out, ref_out)
    return result_row(
        args,
        kernel,
        spec["reference_name"],
        "forward",
        "output",
        status_from_error(stats, args.forward_atol, args.forward_rtol),
        stats,
        notes=spec.get("notes", ""),
    ), ref_out


def run_backward(kernel, spec, inputs, grad_out, args):
    ref_inputs = clone_for_grad(inputs)
    tested_inputs = clone_for_grad(inputs)
    ref_out = spec["reference"](ref_inputs, args)
    tested_out = spec["tested"](tested_inputs, args)
    torch.autograd.backward(ref_out, grad_tensors=grad_out)
    torch.autograd.backward(tested_out, grad_tensors=grad_out)

    rows = []
    for name in spec["backward_inputs"]:
        ref_grad = ref_inputs[name].grad
        tested_grad = tested_inputs[name].grad
        if ref_grad is None or tested_grad is None:
            rows.append(
                base_row(
                    args,
                    kernel,
                    spec["reference_name"],
                    "backward",
                    name,
                    "missing_grad",
                    f"reference_grad={ref_grad is not None}, tested_grad={tested_grad is not None}",
                )
            )
            continue
        stats = error_stats(tested_grad, ref_grad)
        rows.append(
            result_row(
                args,
                kernel,
                spec["reference_name"],
                "backward",
                name,
                status_from_error(stats, args.backward_atol, args.backward_rtol),
                stats,
                notes=spec.get("notes", ""),
            )
        )
    return rows


def run_case(args):
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    inputs = make_inputs(args.batch, args.seqlen, args.nheads, args.headdim, args.headdim, dtype, "cuda")
    rows = []

    for kernel in parse_kernel_list(args.kernels):
        if kernel not in CASE_SPECS:
            raise ValueError(f"Unknown kernel {kernel!r}; choose from {sorted(CASE_SPECS)}")
        spec = CASE_SPECS[kernel]
        try:
            forward_row, ref_out = run_forward(kernel, spec, inputs, args)
            rows.append(forward_row)
        except Exception as exc:
            rows.append(
                base_row(
                    args,
                    kernel,
                    spec["reference_name"],
                    "forward",
                    "output",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if not spec["supports_backward"]:
            rows.append(
                base_row(
                    args,
                    kernel,
                    spec["reference_name"],
                    "backward",
                    "all",
                    "not_applicable",
                    "prototype improved kernel is forward-only",
                )
            )
            continue

        try:
            grad_out = torch.randn_like(ref_out)
            rows.extend(run_backward(kernel, spec, inputs, grad_out, args))
        except Exception as exc:
            rows.append(
                base_row(
                    args,
                    kernel,
                    spec["reference_name"],
                    "backward",
                    "all",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_error(row, field):
    value = row.get(field, "")
    if value == "":
        return ""
    return f"{float(value):.4e}"


def write_markdown(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Kernel | Check | Tensor | Status | Max abs err | Mean abs err | Max rel err | Mean rel err | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['kernel']} | {row['check']} | {row['tensor']} | {row['status']} | "
            f"{format_error(row, 'max_abs_error')} | {format_error(row, 'mean_abs_error')} | "
            f"{format_error(row, 'max_rel_error')} | {format_error(row, 'mean_rel_error')} | "
            f"{row.get('notes', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def as_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_correctness(path, rows):
    numeric_rows = [row for row in rows if as_float(row.get("max_abs_error")) is not None]
    if not numeric_rows:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    numeric_rows = sorted(numeric_rows, key=lambda row: as_float(row["max_abs_error"]) or 0.0, reverse=True)[:32]
    labels = [f"{row['kernel']}\n{row['check']}:{row['tensor']}" for row in numeric_rows]
    values = [max(as_float(row["max_abs_error"]) or 0.0, 1e-12) for row in numeric_rows]
    colors = ["#16a34a" if row["status"] == "pass" else "#dc2626" for row in numeric_rows]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 5))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("max abs error (log scale)")
    ax.set_title("Kernel Correctness vs PyTorch Reference")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build Triton-vs-PyTorch correctness tables.")
    parser.add_argument("--kernels", default="mamba3,simamba,improved")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=8)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--forward-atol", type=float, default=5e-2)
    parser.add_argument("--forward-rtol", type=float, default=5e-2)
    parser.add_argument("--backward-atol", type=float, default=1e-2)
    parser.add_argument("--backward-rtol", type=float, default=1e-2)
    parser.add_argument("--out", default="results/kernel_correctness_reference.csv")
    parser.add_argument("--md-out", default="results/kernel_correctness_reference.md")
    parser.add_argument("--plot-out", default="results/kernel_correctness_reference.png")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="kernel_correctness")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton correctness checks")

    rows = run_case(args)
    out_path = ROOT / args.out
    md_path = ROOT / args.md_out
    plot_path = ROOT / args.plot_out if args.plot_out else None
    write_csv(out_path, rows)
    write_markdown(md_path, rows)
    if plot_path is not None:
        plot_correctness(plot_path, rows)
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
        if plot_path is not None and plot_path.exists():
            wandb.log({"kernel_correctness_error_plot": wandb.Image(str(plot_path))})
        artifact = wandb.Artifact("kernel_correctness_reference", type="profile_results")
        artifact.add_file(str(out_path))
        artifact.add_file(str(md_path))
        if plot_path is not None and plot_path.exists():
            artifact.add_file(str(plot_path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
