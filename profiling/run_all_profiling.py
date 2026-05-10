import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

FINAL_STEPS = (
    "kernel-nsys",
    "kernel-ncu",
    "kernel-correctness",
    "vllm",
    "vllm-combine",
    "bundle",
)
STEP_ALIASES = {
    "nsys": "kernel-nsys",
    "ncu": "kernel-ncu",
    "correctness": "kernel-correctness",
    "combine": "vllm-combine",
}
ALL_STEPS = set(FINAL_STEPS) | set(STEP_ALIASES)


def result_path(args, *parts):
    return str(Path(args.results_dir, *parts))


def repo_result_path(args, *parts):
    path = Path(args.results_dir, *parts)
    if path.is_absolute():
        return str(path)
    return str(Path("profiling") / path)


def script(name):
    return str(Path("profiling") / name)


def maybe_wandb(args):
    return ["--wandb"] if args.wandb else []


def parse_steps(value):
    requested = [item.strip().lower().replace("_", "-") for item in value.split(",") if item.strip()]
    if not requested or "all" in requested:
        return list(FINAL_STEPS)
    normalized = [STEP_ALIASES.get(item, item) for item in requested]
    unknown = sorted(set(normalized) - set(FINAL_STEPS))
    if unknown:
        raise ValueError(f"unknown profiling step(s): {', '.join(unknown)}")
    return normalized


def add_if(cmd, condition, *items):
    if condition:
        cmd.extend(items)
    return cmd


def run_command(args, name, cmd, env):
    print(f"\n==> {name}", flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    except subprocess.CalledProcessError:
        if args.continue_on_error:
            print(f"warning: step failed and --continue-on-error is set: {name}", flush=True)
            return
        raise


def kernel_nsys_command(args):
    return [
        args.python,
        script("nsys_kernel_profile.py"),
        "--kernels",
        args.kernel_kernels,
        "--batch",
        str(args.kernel_batch),
        "--seqlen",
        str(args.kernel_seqlen),
        "--nheads",
        str(args.kernel_nheads),
        "--headdim",
        str(args.kernel_headdim),
        "--warmup",
        str(args.kernel_warmup),
        "--dtype",
        args.kernel_dtype,
        "--chunk-size",
        str(args.kernel_chunk_size),
        "--num-warps",
        str(args.kernel_num_warps),
        "--num-stages",
        str(args.kernel_num_stages),
        "--out-dir",
        result_path(args, "kernel_suite", "nsys"),
        "--stats-out-dir",
        "csv_exports",
        "--nsys-bin",
        args.nsys_bin,
        "--wandb-group",
        "kernel_suite_nsys",
        "--wandb-name",
        f"kernel_suite_nsys_b{args.kernel_batch}_s{args.kernel_seqlen}",
        *maybe_wandb(args),
    ]


def kernel_ncu_command(args):
    cmd = [
        args.python,
        script("ncu_kernel_profile.py"),
        "--kernel-targets",
        args.ncu_kernel_targets,
        "--seq-lens",
        args.ncu_seq_lens,
        "--batch-sizes",
        args.ncu_batch_sizes,
        "--nheads",
        str(args.kernel_nheads),
        "--headdim",
        str(args.kernel_headdim),
        "--warmup",
        str(args.kernel_warmup),
        "--dtype",
        args.kernel_dtype,
        "--chunk-size",
        str(args.kernel_chunk_size),
        "--num-warps",
        str(args.kernel_num_warps),
        "--num-stages",
        str(args.kernel_num_stages),
        "--out-dir",
        result_path(args, "kernel_suite", "ncu"),
        "--ncu-bin",
        args.ncu_bin,
        "--set",
        args.ncu_set,
        "--launch-count",
        str(args.ncu_launch_count),
        "--kernel-name-base",
        args.ncu_kernel_name_base,
        "--wandb-group",
        "kernel_suite_ncu",
        "--wandb-name",
        "kernel_suite_ncu_filtered_targets",
        *maybe_wandb(args),
    ]
    add_if(cmd, bool(args.ncu_page), "--page", args.ncu_page)
    add_if(cmd, args.sudo_ncu, "--sudo-ncu")
    return cmd


def kernel_correctness_command(args):
    return [
        args.python,
        script("kernel_correctness.py"),
        "--kernels",
        args.kernel_kernels,
        "--batch",
        str(args.correctness_batch),
        "--seqlen",
        str(args.correctness_seqlen),
        "--nheads",
        str(args.correctness_nheads),
        "--headdim",
        str(args.correctness_headdim),
        "--dtype",
        args.correctness_dtype,
        "--chunk-size",
        str(args.kernel_chunk_size),
        "--num-warps",
        str(args.kernel_num_warps),
        "--num-stages",
        str(args.kernel_num_stages),
        "--out",
        result_path(args, "kernel_suite", "kernel_correctness.csv"),
        "--md-out",
        result_path(args, "kernel_suite", "kernel_correctness.md"),
        "--plot-out",
        result_path(args, "kernel_suite", "kernel_correctness.png"),
        "--wandb-group",
        "kernel_suite_correctness",
        "--wandb-name",
        "kernel_suite_correctness_triton_vs_pytorch",
        *maybe_wandb(args),
    ]


def common_vllm_args(args, model, stem, wandb_name):
    cmd = [
        args.python,
        script("vllm_sweep.py"),
        "--models",
        model,
        "--prompt-words",
        args.prompt_words,
        "--batch-sizes",
        args.batch_sizes,
        "--max-tokens",
        args.max_tokens,
        "--prefix-cache-modes",
        args.prefix_cache_modes,
        "--repeated-prefix-tokens",
        str(args.repeated_prefix_tokens),
        "--mamba-block-size",
        str(args.mamba_block_size),
        "--warmup",
        str(args.vllm_warmup),
        "--repeats",
        str(args.vllm_repeats),
        "--prefill-probe-tokens",
        str(args.prefill_probe_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cuda-device",
        str(args.cuda_device),
        "--memory-poll-interval-sec",
        str(args.memory_poll_interval_sec),
        "--trust-remote-code",
        "--out",
        result_path(args, f"{stem}_summary.csv"),
        "--raw-out",
        result_path(args, f"{stem}_raw.csv"),
        "--plot-out",
        result_path(args, f"{stem}.png"),
        "--wandb-group",
        "model_vllm_profile",
        "--wandb-name",
        wandb_name,
        *maybe_wandb(args),
    ]
    add_if(cmd, not args.no_cudagraph, "--use-cudagraph")
    return cmd


def vllm_mamba2_command(args):
    return common_vllm_args(args, args.mamba2_model, "vllm_mamba2", "vllm_mamba2_ttft_tpot_decode_cache")


def vllm_improved_command(args):
    cmd = common_vllm_args(
        args,
        args.improved_model,
        "vllm_improved_simamba",
        "vllm_improved_simamba_ttft_tpot_decode_cache",
    )
    cmd.extend(["--model-impl", args.improved_model_impl])
    add_if(cmd, bool(args.simamba_vllm_fork), "--simamba-vllm-fork", args.simamba_vllm_fork)
    add_if(cmd, bool(args.simamba_native_backend), "--simamba-native-backend", args.simamba_native_backend)
    return cmd


def vllm_combine_command(args):
    return [
        args.python,
        script("vllm_combine_results.py"),
        "--summary-csvs",
        result_path(args, "vllm_mamba2_summary.csv"),
        result_path(args, "vllm_improved_simamba_summary.csv"),
        "--raw-csvs",
        result_path(args, "vllm_mamba2_raw.csv"),
        result_path(args, "vllm_improved_simamba_raw.csv"),
        "--out",
        result_path(args, "vllm_mamba2_vs_improved_summary.csv"),
        "--raw-out",
        result_path(args, "vllm_mamba2_vs_improved_raw.csv"),
        "--plot-out",
        result_path(args, "vllm_mamba2_vs_improved.png"),
        "--wandb-group",
        "model_vllm_profile",
        "--wandb-name",
        "vllm_mamba2_vs_improved_combined",
        *maybe_wandb(args),
    ]


def bundle_command(args):
    paths = [
        repo_result_path(args, "kernel_suite", "nsys"),
        repo_result_path(args, "kernel_suite", "ncu"),
        repo_result_path(args, "kernel_suite", "kernel_correctness.csv"),
        repo_result_path(args, "kernel_suite", "kernel_correctness.md"),
        repo_result_path(args, "kernel_suite", "kernel_correctness.png"),
        repo_result_path(args, "vllm_mamba2_summary.csv"),
        repo_result_path(args, "vllm_mamba2_raw.csv"),
        repo_result_path(args, "vllm_mamba2.png"),
        repo_result_path(args, "vllm_improved_simamba_summary.csv"),
        repo_result_path(args, "vllm_improved_simamba_raw.csv"),
        repo_result_path(args, "vllm_improved_simamba.png"),
        repo_result_path(args, "vllm_mamba2_vs_improved_summary.csv"),
        repo_result_path(args, "vllm_mamba2_vs_improved_raw.csv"),
        repo_result_path(args, "vllm_mamba2_vs_improved.png"),
    ]
    return [
        args.python,
        script("wandb_log_bundle.py"),
        "--group",
        "profiling_orchestration",
        "--run-name",
        "profiling_orchestration_bundle",
        "--name",
        "profiling_orchestration_bundle",
        "--type",
        "profile_results",
        "--paths",
        *paths,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run kernel and vLLM profiling orchestration.")
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated steps: all, kernel-nsys, kernel-ncu, kernel-correctness, "
            "vllm, vllm-combine, bundle. Short aliases: nsys, ncu, correctness, combine."
        ),
    )
    parser.add_argument("--results-dir", default="results", help="Output directory relative to profiling/ unless absolute.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb", action="store_true", help="Log each step to W&B project profiling.")
    parser.add_argument("--wandb-project", default="profiling", help="W&B project to force for profiling runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--kernel-kernels", default="mamba3,simamba,improved")
    parser.add_argument("--kernel-batch", type=int, default=2)
    parser.add_argument("--kernel-seqlen", type=int, default=256)
    parser.add_argument("--kernel-nheads", type=int, default=32)
    parser.add_argument("--kernel-headdim", type=int, default=64)
    parser.add_argument("--kernel-warmup", type=int, default=5)
    parser.add_argument("--kernel-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--kernel-chunk-size", type=int, default=64)
    parser.add_argument("--kernel-num-warps", type=int, default=8)
    parser.add_argument("--kernel-num-stages", type=int, default=3)
    parser.add_argument("--nsys-bin", default="/usr/local/cuda/bin/nsys")

    parser.add_argument(
        "--ncu-kernel-targets",
        default="mamba3_siso_fwd_kernel,simamba_siso_fwd_kernel,improved_siso_prefill_kernel",
    )
    parser.add_argument("--ncu-seq-lens", default="256")
    parser.add_argument("--ncu-batch-sizes", default="2")
    parser.add_argument("--ncu-bin", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--ncu-set", default="full")
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    parser.add_argument("--ncu-kernel-name-base", default="demangled")
    parser.add_argument("--ncu-page", default="raw")
    parser.add_argument("--sudo-ncu", action="store_true", help="Run Nsight Compute through sudo -n -E.")

    parser.add_argument("--correctness-batch", type=int, default=2)
    parser.add_argument("--correctness-seqlen", type=int, default=8)
    parser.add_argument("--correctness-nheads", type=int, default=4)
    parser.add_argument("--correctness-headdim", type=int, default=16)
    parser.add_argument("--correctness-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")

    parser.add_argument("--mamba2-model", default="soumil1/mamba2-10m-slimpajama-500m")
    parser.add_argument(
        "--improved-model",
        "--simamba-model",
        dest="improved_model",
        default="outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export",
    )
    parser.add_argument("--simamba-vllm-fork", default="/tmp/hpmls26_vllm")
    parser.add_argument("--simamba-native-backend", default="triton")
    parser.add_argument("--improved-model-impl", default="auto")
    parser.add_argument("--prompt-words", default="128")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--max-tokens", default="32")
    parser.add_argument("--prefix-cache-modes", default="off,on")
    parser.add_argument("--repeated-prefix-tokens", type=int, default=512)
    parser.add_argument("--mamba-block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--vllm-warmup", type=int, default=1)
    parser.add_argument("--vllm-repeats", type=int, default=5)
    parser.add_argument("--prefill-probe-tokens", type=int, default=1)
    parser.add_argument("--memory-poll-interval-sec", type=float, default=0.005)
    parser.add_argument("--no-cudagraph", action="store_true")

    args = parser.parse_args()
    steps = parse_steps(args.steps)

    env = os.environ.copy()
    if args.wandb:
        env["WANDB_PROJECT"] = args.wandb_project

    commands = []
    if "kernel-nsys" in steps:
        commands.append(("kernel-nsys", kernel_nsys_command(args)))
    if "kernel-ncu" in steps:
        commands.append(("kernel-ncu", kernel_ncu_command(args)))
    if "kernel-correctness" in steps:
        commands.append(("kernel-correctness", kernel_correctness_command(args)))
    if "vllm" in steps:
        commands.extend(
            [
                ("vllm-mamba2", vllm_mamba2_command(args)),
                ("vllm-improved-simamba", vllm_improved_command(args)),
            ]
        )
    if "vllm-combine" in steps:
        commands.append(("vllm-combine", vllm_combine_command(args)))
    if "bundle" in steps:
        if args.wandb:
            commands.append(("wandb-bundle", bundle_command(args)))
        else:
            print("Skipping bundle step because --wandb is not set.", flush=True)

    for name, cmd in commands:
        run_command(args, name, cmd, env)


if __name__ == "__main__":
    main()
