import argparse
import os
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description="Upload profiling result files to W&B as one artifact.")
    parser.add_argument("--name", default="profiling_report_bundle")
    parser.add_argument("--type", default="profile_results")
    parser.add_argument("--group", default="profiling_full_run")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "profiling"),
        job_type="profiling_bundle",
        group=args.group,
        name=args.run_name,
        config=vars(args),
    )
    artifact = wandb.Artifact(args.name, type=args.type)
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            artifact.add_dir(str(path))
        elif path.exists():
            artifact.add_file(str(path))
        else:
            wandb.log({f"missing/{item}": 1})
            print(f"missing: {item}", flush=True)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
