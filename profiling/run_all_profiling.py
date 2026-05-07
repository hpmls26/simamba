import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


FINAL_STEPS = ("nsys", "vllm", "combine", "correctness", "bundle")
OPTIONAL_STEPS = ("kernel-sweep", "plots")
ALL_STEPS = FINAL_STEPS + OPTIONAL_STEPS


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


def nsys_stem(kernel, args):
    return f"{kernel}_b{args.nsys_batch}_s{args.nsys_seqlen}_h{args.nsys_nheads}_d{args.nsys_headdim}"


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


def parse_steps(value):
    requested = [item.strip().lower().replace("_", "-") for item in value.split(",") if item.strip()]
    if not requested or "all" in requested:
        return list(FINAL_STEPS)
    unknown = sorted(set(requested) - set(ALL_STEPS))
    if unknown:
        raise ValueError(f"unknown profiling step(s): {', '.join(unknown)}")
    return requested


def nsys_command(args):
    return [
        args.python,
        script("nsys_kernel_profile.py"),
        "--kernels",
        args.nsys_kernels,
        "--batch",
        str(args.nsys_batch),
        "--seqlen",
        str(args.nsys_seqlen),
        "--nheads",
        str(args.nsys_nheads),
        "--headdim",
        str(args.nsys_headdim),
        "--warmup",
        str(args.nsys_warmup),
        "--out-dir",
        result_path(args, "nsys_task_named_wandb"),
        "--nsys-bin",
        args.nsys_bin,
        "--wandb-group",
        "task_1_nsys_prefill_baselines",
        "--wandb-name",
        f"task_1_nsys_simamba_mamba3_prefill_b{args.nsys_batch}_s{args.nsys_seqlen}",
        *maybe_wandb(args),
    ]


def vllm_mamba2_command(args):
    return [
        args.python,
        script("vllm_sweep.py"),
        "--models",
        args.mamba2_model,
        "--prompt-words",
        str(args.prompt_words),
        "--batch-sizes",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--prefix-cache-modes",
        "off,on",
        "--repeated-prefix-tokens",
        str(args.repeated_prefix_tokens),
        "--mamba-block-size",
        str(args.mamba_block_size),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cuda-device",
        str(args.cuda_device),
        "--out",
        result_path(args, "vllm_mamba2_repeated_summary.csv"),
        "--raw-out",
        result_path(args, "vllm_mamba2_repeated_raw.csv"),
        "--plot-out",
        result_path(args, "vllm_mamba2_repeated_summary.png"),
        "--wandb-group",
        "task_3_vllm_repeated_profile_fixed",
        "--wandb-name",
        "task_3_vllm_mamba2_repeated_cache_off_on",
        *maybe_wandb(args),
    ]


def vllm_simamba_command(args, prefix_cache):
    stem = f"vllm_simamba_repeated_{prefix_cache}"
    cmd = [
        args.python,
        script("vllm_sweep.py"),
        "--models",
        args.simamba_model,
        "--prompt-words",
        str(args.prompt_words),
        "--batch-sizes",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--prefix-cache-modes",
        prefix_cache,
        "--repeated-prefix-tokens",
        str(args.repeated_prefix_tokens),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cuda-device",
        str(args.cuda_device),
        "--trust-remote-code",
        "--model-impl",
        "transformers",
        "--force-transformers-backend-compatible",
        "--out",
        result_path(args, f"{stem}_summary.csv"),
        "--raw-out",
        result_path(args, f"{stem}_raw.csv"),
        "--plot-out",
        result_path(args, f"{stem}_summary.png"),
        "--wandb-group",
        "task_3_vllm_repeated_profile_fixed",
        "--wandb-name",
        f"task_3_vllm_simamba_repeated_cache_{prefix_cache}",
        *maybe_wandb(args),
    ]
    if prefix_cache == "on":
        cmd.extend(["--mamba-block-size", str(args.mamba_block_size)])
    return cmd


def combine_command(args):
    return [
        args.python,
        script("vllm_combine_results.py"),
        "--summary-csvs",
        result_path(args, "vllm_mamba2_repeated_summary.csv"),
        result_path(args, "vllm_simamba_repeated_off_summary.csv"),
        result_path(args, "vllm_simamba_repeated_on_summary.csv"),
        "--raw-csvs",
        result_path(args, "vllm_mamba2_repeated_raw.csv"),
        result_path(args, "vllm_simamba_repeated_off_raw.csv"),
        result_path(args, "vllm_simamba_repeated_on_raw.csv"),
        "--out",
        result_path(args, "vllm_repeated_combined_summary.csv"),
        "--raw-out",
        result_path(args, "vllm_repeated_combined_raw.csv"),
        "--plot-out",
        result_path(args, "vllm_repeated_combined_summary.png"),
        "--wandb-group",
        "task_3_vllm_repeated_profile_fixed",
        "--wandb-name",
        "task_3_vllm_combined_repeated_profile_fixed",
        *maybe_wandb(args),
    ]


def correctness_command(args):
    return [
        args.python,
        script("kernel_correctness.py"),
        "--out",
        result_path(args, "kernel_correctness_reference.csv"),
        "--md-out",
        result_path(args, "kernel_correctness_reference.md"),
        "--wandb-group",
        "task_5_correctness_triton_vs_pytorch_reference",
        "--wandb-name",
        "task_5_correctness_forward_backward_reference",
        *maybe_wandb(args),
    ]


def kernel_sweep_command(args):
    return [
        args.python,
        script("kernel_sweep.py"),
        "--seq-lens",
        args.kernel_seq_lens,
        "--batch-sizes",
        args.kernel_batch_sizes,
        "--iters",
        str(args.kernel_iters),
        "--out",
        result_path(args, "kernel_sweep.csv"),
        "--wandb-group",
        "task_optional_kernel_sweep",
        "--wandb-name",
        "task_optional_kernel_latency_memory_sweep",
        "--compare",
        *maybe_wandb(args),
    ]


def plots_command(args):
    return [
        args.python,
        script("plot_profile_results.py"),
        "--kernel-csv",
        result_path(args, "kernel_sweep.csv"),
        "--vllm-csv",
        result_path(args, "vllm_repeated_combined_summary.csv"),
        "--out-dir",
        result_path(args, "plots"),
        *maybe_wandb(args),
    ]


def bundle_command(args):
    paths = [
        repo_result_path(args, "nsys_task_named_wandb", f"{nsys_stem('mamba3', args)}.nsys-rep"),
        repo_result_path(args, "nsys_task_named_wandb", f"{nsys_stem('simamba', args)}.nsys-rep"),
        repo_result_path(args, "nsys_task_named_wandb", "nsys_exports"),
        repo_result_path(args, "vllm_mamba2_repeated_summary.csv"),
        repo_result_path(args, "vllm_mamba2_repeated_raw.csv"),
        repo_result_path(args, "vllm_mamba2_repeated_summary.png"),
        repo_result_path(args, "vllm_simamba_repeated_off_summary.csv"),
        repo_result_path(args, "vllm_simamba_repeated_off_raw.csv"),
        repo_result_path(args, "vllm_simamba_repeated_off_summary.png"),
        repo_result_path(args, "vllm_simamba_repeated_on_summary.csv"),
        repo_result_path(args, "vllm_simamba_repeated_on_raw.csv"),
        repo_result_path(args, "vllm_simamba_repeated_on_summary.png"),
        repo_result_path(args, "vllm_repeated_combined_summary.csv"),
        repo_result_path(args, "vllm_repeated_combined_raw.csv"),
        repo_result_path(args, "vllm_repeated_combined_summary.png"),
        repo_result_path(args, "kernel_correctness_reference.csv"),
        repo_result_path(args, "kernel_correctness_reference.md"),
    ]
    return [
        args.python,
        script("wandb_log_bundle.py"),
        "--group",
        "profiling_full_run",
        "--run-name",
        "profiling_full_run_bundle",
        "--name",
        "profiling_full_run_bundle",
        "--type",
        "profile_results",
        "--paths",
        *paths,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the final Simamba profiling reproduction pipeline.")
    parser.add_argument("--steps", default="all", help="Comma-separated steps: all, nsys, vllm, combine, correctness, bundle, kernel-sweep, plots.")
    parser.add_argument("--results-dir", default="results", help="Output directory relative to profiling/ unless absolute.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb", action="store_true", help="Log each step to W&B project profiling.")
    parser.add_argument("--wandb-project", default="profiling", help="W&B project to force for profiling runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--nsys-bin", default="/usr/local/cuda/bin/nsys")
    parser.add_argument("--nsys-kernels", default="simamba,mamba3")
    parser.add_argument("--nsys-batch", type=int, default=2)
    parser.add_argument("--nsys-seqlen", type=int, default=256)
    parser.add_argument("--nsys-nheads", type=int, default=32)
    parser.add_argument("--nsys-headdim", type=int, default=64)
    parser.add_argument("--nsys-warmup", type=int, default=5)

    parser.add_argument("--mamba2-model", default="soumil1/mamba2-10m-slimpajama-500m")
    parser.add_argument("--simamba-model", default="soumil1/simamba-midpoint-10m-slimpajama-500m")
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeated-prefix-tokens", type=int, default=512)
    parser.add_argument("--mamba-block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)

    parser.add_argument("--kernel-seq-lens", default="128,256,512,1024")
    parser.add_argument("--kernel-batch-sizes", default="1,2,4")
    parser.add_argument("--kernel-iters", type=int, default=20)

    args = parser.parse_args()
    steps = parse_steps(args.steps)

    env = os.environ.copy()
    if args.wandb:
        env["WANDB_PROJECT"] = args.wandb_project

    commands = []
    if "nsys" in steps:
        commands.append(("nsys", nsys_command(args)))
    if "vllm" in steps:
        commands.extend(
            [
                ("vllm-mamba2", vllm_mamba2_command(args)),
                ("vllm-simamba-cache-off", vllm_simamba_command(args, "off")),
                ("vllm-simamba-cache-on", vllm_simamba_command(args, "on")),
            ]
        )
    if "combine" in steps:
        commands.append(("vllm-combine", combine_command(args)))
    if "correctness" in steps:
        commands.append(("correctness", correctness_command(args)))
    if "kernel-sweep" in steps:
        commands.append(("kernel-sweep", kernel_sweep_command(args)))
    if "plots" in steps:
        commands.append(("plots", plots_command(args)))
    if "bundle" in steps:
        if args.wandb:
            commands.append(("wandb-bundle", bundle_command(args)))
        else:
            print("Skipping bundle step because --wandb is not set.", flush=True)

    for name, cmd in commands:
        run_command(args, name, cmd, env)


if __name__ == "__main__":
    main()
