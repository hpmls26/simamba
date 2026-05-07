import argparse
import os
import subprocess
import sys
from pathlib import Path

import wandb


KERNEL_DIRS = {
    "simamba": "simamba",
    "mamba3": "mamba3",
}


def main():
    parser = argparse.ArgumentParser(description="Collect Nsight Systems reports for kernel profilers.")
    parser.add_argument("--kernels", default="simamba,mamba3")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out-dir", default="results/nsys")
    parser.add_argument("--nsys-bin", default="/usr/local/cuda/bin/nsys")
    parser.add_argument("--stats-reports", default="cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,osrt_sum")
    parser.add_argument("--stats-out-dir", default="nsys_exports")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="nsys_kernel_profile")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="nsys_kernel_profile",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )

    produced = []
    stats_files = []
    for kernel in [item for item in args.kernels.split(",") if item]:
        workdir = root / KERNEL_DIRS[kernel]
        stem = f"{kernel}_b{args.batch}_s{args.seqlen}_h{args.nheads}_d{args.headdim}"
        out_stem = out_dir / stem
        env = os.environ.copy()
        env.update({
            "DISABLE_KINETO": "1",
            "USE_TORCH_PROFILER": "0",
            "PROFILE_BATCH": str(args.batch),
            "PROFILE_SEQLEN": str(args.seqlen),
            "PROFILE_NHEADS": str(args.nheads),
            "PROFILE_HEADDIM": str(args.headdim),
            "PROFILE_WARMUP": str(args.warmup),
        })
        cmd = [
            args.nsys_bin,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi",
            f"--output={out_stem}",
            "--force-overwrite=true",
            sys.executable,
            "kernel_profiler.py",
        ]
        print(" ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=workdir, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"nsys exited with {exc.returncode}; checking for generated report.", flush=True)
        report = out_stem.with_suffix(".nsys-rep")
        if not report.exists():
            raise RuntimeError(f"NSYS report was not generated: {report}")
        produced.append(report)
        stats_dir = out_dir / args.stats_out_dir
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_prefix = stats_dir / f"{kernel}_stats"
        stats_cmd = [
            args.nsys_bin,
            "stats",
            "--report",
            args.stats_reports,
            "--format",
            "csv",
            "--output",
            str(stats_prefix),
            str(report),
        ]
        print(" ".join(stats_cmd), flush=True)
        subprocess.run(stats_cmd, check=True)
        stats_files.extend(sorted(stats_dir.glob(f"{kernel}_stats_*.csv")))
        if run is not None:
            wandb.log({
                "kernel": kernel,
                "batch": args.batch,
                "seqlen": args.seqlen,
                "nheads": args.nheads,
                "headdim": args.headdim,
            })

    if run is not None:
        artifact = wandb.Artifact("nsys_kernel_reports", type="nsys_reports")
        for report in produced:
            if report.exists():
                artifact.add_file(str(report))
        for stats_file in stats_files:
            if stats_file.exists():
                artifact.add_file(str(stats_file))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
