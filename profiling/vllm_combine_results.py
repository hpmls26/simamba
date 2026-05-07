import argparse
import csv
import os
from pathlib import Path

import wandb

from vllm_sweep import fieldnames_for, plot_summary, wandb_table, write_csv


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description="Combine vLLM profile CSVs and log one comparison plot.")
    parser.add_argument("--summary-csvs", nargs="+", required=True)
    parser.add_argument("--raw-csvs", nargs="+", required=True)
    parser.add_argument("--out", default="results/vllm_combined_summary.csv")
    parser.add_argument("--raw-out", default="results/vllm_combined_raw.csv")
    parser.add_argument("--plot-out", default="results/vllm_combined_summary.png")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="task_3_vllm_repeated_profile_fixed")
    parser.add_argument("--wandb-name", default="task_3_vllm_combined_repeated_profile")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    summary_rows = []
    raw_rows = []
    for path in args.summary_csvs:
        summary_rows.extend(read_csv(root / path))
    for path in args.raw_csvs:
        raw_rows.extend(read_csv(root / path))

    summary_path = root / args.out
    raw_path = root / args.raw_out
    plot_path = root / args.plot_out
    write_csv(summary_path, summary_rows)
    write_csv(raw_path, raw_rows)
    plot_summary(summary_rows, plot_path)

    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="vllm_combined_profile",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )
        summary_table = wandb_table(summary_rows)
        raw_table = wandb_table(raw_rows)
        if summary_table is not None:
            wandb.log({"vllm_combined_summary_table": summary_table})
        if raw_table is not None:
            wandb.log({"vllm_combined_raw_samples_table": raw_table})
        if plot_path.exists():
            wandb.log({"vllm_combined_summary_plot": wandb.Image(str(plot_path))})
        artifact = wandb.Artifact("vllm_combined_repeated_profile", type="profile_results")
        artifact.add_file(str(summary_path))
        artifact.add_file(str(raw_path))
        if plot_path.exists():
            artifact.add_file(str(plot_path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
