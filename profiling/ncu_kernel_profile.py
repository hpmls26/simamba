import argparse
import os
import subprocess
import sys
from pathlib import Path

import wandb


KERNEL_DIRS = {
    "simamba": "simamba",
    "mamba3": "mamba3",
    "improved": "test_kernel",
}

KERNEL_FILTERS = {
    "simamba_mul_kernel": ("simamba", "regex:.*MulFunctor<float>.*"),
    "simamba_exp_kernel": ("simamba", "regex:.*exp_kernel_cuda.*"),
    "simamba_gemv_kernel": ("simamba", "regex:.*gemvx::kernel.*"),
    "mamba3_siso_fwd_kernel": ("mamba3", "regex:.*mamba3_siso_fwd_kernel.*"),
    "simamba_siso_fwd_kernel": ("simamba", "regex:.*mamba3_siso_fwd_kernel.*"),
    "improved_siso_prefill_kernel": ("improved", "regex:.*simamba_siso_prefill_chunk_kernel.*"),
}


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def run_command(cmd, cwd=None, env=None, stdout_path=None):
    if stdout_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    with stdout_path.open("w") as handle:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def export_existing_report(args, out_dir):
    report = Path(args.import_report)
    if not report.is_absolute():
        report = Path.cwd() / report
    if not report.exists():
        raise FileNotFoundError(report)
    stem = args.import_stem or report.with_suffix("").name
    raw_csv = out_dir / f"{stem}_raw.csv"
    details_txt = out_dir / f"{stem}_details.txt"
    raw_cmd = [args.ncu_bin, "--import", str(report), "--csv", "--page", "raw"]
    details_cmd = [args.ncu_bin, "--import", str(report), "--page", "details"]
    print(" ".join(raw_cmd) + f" > {raw_csv}", flush=True)
    run_command(raw_cmd, stdout_path=raw_csv)
    print(" ".join(details_cmd) + f" > {details_txt}", flush=True)
    run_command(details_cmd, stdout_path=details_txt)


def export_report_pages(ncu_bin, report, out_dir, stem):
    raw_csv = out_dir / f"{stem}_import_raw.csv"
    details_txt = out_dir / f"{stem}_details.txt"
    raw_cmd = [ncu_bin, "--import", str(report), "--csv", "--page", "raw"]
    details_cmd = [ncu_bin, "--import", str(report), "--page", "details"]
    print(" ".join(raw_cmd) + f" > {raw_csv}", flush=True)
    run_command(raw_cmd, stdout_path=raw_csv)
    print(" ".join(details_cmd) + f" > {details_txt}", flush=True)
    run_command(details_cmd, stdout_path=details_txt)


def main():
    parser = argparse.ArgumentParser(description="Collect Nsight Compute reports for kernel shapes.")
    parser.add_argument("--kernels", default="mamba3,simamba,improved")
    parser.add_argument(
        "--kernel-targets",
        default="mamba3_siso_fwd_kernel,simamba_siso_fwd_kernel,improved_siso_prefill_kernel",
    )
    parser.add_argument("--seq-lens", default="256,1024")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--out-dir", default="results/ncu")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--sudo-ncu", action="store_true", help="Run Nsight Compute through sudo -n.")
    parser.add_argument("--set", default="full")
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--kernel-name-base", default="demangled")
    parser.add_argument("--page", default="")
    parser.add_argument("--import-report", default="")
    parser.add_argument("--import-stem", default="")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="ncu_kernel_profile")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.import_report:
        export_existing_report(args, out_dir)
        if args.import_only:
            return
    run = None
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="ncu_kernel_profile",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )

    targets = []
    for target in [item for item in args.kernel_targets.split(",") if item]:
        if target not in KERNEL_FILTERS:
            raise ValueError(f"Unknown kernel target {target!r}; choose from {sorted(KERNEL_FILTERS)}")
        targets.append((target, *KERNEL_FILTERS[target]))
    if not targets:
        targets = [(kernel, kernel, "") for kernel in [item for item in args.kernels.split(",") if item]]

    for target_name, kernel, kernel_name_filter in targets:
        if kernel not in KERNEL_DIRS:
            raise ValueError(f"Unknown kernel {kernel!r}; choose from {sorted(KERNEL_DIRS)}")
        workdir = root / KERNEL_DIRS[kernel]
        for batch in parse_ints(args.batch_sizes):
            for seqlen in parse_ints(args.seq_lens):
                stem = f"{target_name}_b{batch}_s{seqlen}_h{args.nheads}_d{args.headdim}"
                env = os.environ.copy()
                env.update({
                    "PROFILE_BATCH": str(batch),
                    "PROFILE_SEQLEN": str(seqlen),
                    "PROFILE_NHEADS": str(args.nheads),
                    "PROFILE_HEADDIM": str(args.headdim),
                    "PROFILE_WARMUP": str(args.warmup),
                    "PROFILE_DTYPE": args.dtype,
                    "PROFILE_CHUNK_SIZE": str(args.chunk_size),
                    "PROFILE_NUM_WARPS": str(args.num_warps),
                    "PROFILE_NUM_STAGES": str(args.num_stages),
                })
                cmd = [
                    args.ncu_bin,
                    "--set", args.set,
                    "--target-processes", "all",
                    "--profile-from-start", "off",
                ]
                if kernel_name_filter:
                    cmd.extend(["--kernel-name-base", args.kernel_name_base, "--kernel-name", kernel_name_filter])
                cmd.extend([
                    "--launch-count", str(args.launch_count),
                    "--csv",
                ])
                if args.page:
                    cmd.extend(["--page", args.page])
                cmd.extend([
                    "--export", str(out_dir / stem),
                    "--force-overwrite",
                    sys.executable,
                    "kernel_profiler.py",
                ])
                if args.sudo_ncu:
                    cmd = ["sudo", "-n", "-E"] + cmd
                print(" ".join(cmd), flush=True)
                run_command(cmd, cwd=workdir, env=env, stdout_path=out_dir / f"{stem}.csv")
                report = out_dir / f"{stem}.ncu-rep"
                if not report.exists():
                    raise RuntimeError(f"NCU report was not generated: {report}")
                export_report_pages(args.ncu_bin, report, out_dir, stem)
                if run is not None:
                    wandb.log({
                        "target": target_name,
                        "kernel": kernel,
                        "kernel_name_filter": kernel_name_filter,
                        "batch": batch,
                        "seqlen": seqlen,
                        "nheads": args.nheads,
                        "headdim": args.headdim,
                    })

    if run is not None:
        artifact = wandb.Artifact("ncu_kernel_reports", type="ncu_reports")
        artifact.add_dir(str(out_dir))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
