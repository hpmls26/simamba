#!/usr/bin/env python3
"""Upload the best Simamba checkpoint export to Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, get_token, upload_folder


DEFAULT_EXPORT_DIR = Path("hf_exports/simamba-midpoint-10m-slimpajama-500m")
REQUIRED_FILES = (
    "README.md",
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "metrics.json",
    "checkpoint_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination repo id, for example username/simamba-midpoint-10m-slimpajama-500m.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"Prepared HF export directory. Default: {DEFAULT_EXPORT_DIR}",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repo as private.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the export directory without uploading.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload model artifacts",
        help="Hugging Face commit message.",
    )
    return parser.parse_args()


def validate_export(export_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{export_dir} is missing required upload files: {', '.join(missing)}"
        )


def stage_with_repo_id(export_dir: Path, repo_id: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="simamba_hf_upload_"))
    staged = tmpdir / export_dir.name
    shutil.copytree(export_dir, staged)
    readme = staged / "README.md"
    readme.write_text(
        readme.read_text().replace("REPLACE_WITH_HF_REPO_ID", repo_id),
        encoding="utf-8",
    )
    return staged


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    validate_export(export_dir)

    if args.dry_run:
        print(f"validated_export_dir={export_dir}")
        print(f"repo_id={args.repo_id}")
        print("dry_run=true")
        return

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Run "
            "`.venv/bin/python -c \"from huggingface_hub import login; login()\"` "
            "or set `HF_TOKEN`, then retry."
        )

    staged = stage_with_repo_id(export_dir, args.repo_id)
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(staged),
        token=token,
        commit_message=args.commit_message,
    )
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
