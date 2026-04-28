#!/usr/bin/env python

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional


STEP_ARCHIVE_RE = re.compile(r"step_(\d+)\.tar$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch the latest uploaded step_*.tar checkpoint archive from GCS and restore it locally."
    )
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET"), help="Target GCS bucket name.")
    parser.add_argument(
        "--object-prefix",
        default=None,
        help="Object prefix to search under. Defaults to GCS_PREFIX/GCS_RUN_PREFIX/checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory.")
    parser.add_argument(
        "--restore-dir-name",
        default="latest",
        help="Directory name under output-dir to restore into. Defaults to latest.",
    )
    return parser.parse_args()


def new_storage_client():
    from google.cloud import storage

    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return storage.Client(project=project)
    return storage.Client()


def default_object_prefix() -> str:
    parts = [
        os.environ.get("GCS_PREFIX", "").strip("/"),
        os.environ.get("GCS_RUN_PREFIX", "").strip("/"),
        "checkpoints",
    ]
    return "/".join(part for part in parts if part)


def find_latest_step_blob(bucket, prefix: str):
    latest_blob = None
    latest_step = -1
    prefix = prefix.strip("/")
    prefix_arg = f"{prefix}/" if prefix else ""
    for blob in bucket.list_blobs(prefix=prefix_arg):
        name = blob.name.rsplit("/", 1)[-1]
        match = STEP_ARCHIVE_RE.fullmatch(name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_blob = blob
    return latest_blob, latest_step


def safe_extract_tar(archive_path: Path, destination: Path):
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination_resolved / member.name).resolve()
            if not str(member_path).startswith(str(destination_resolved)):
                raise RuntimeError(f"Refusing to extract unsafe member {member.name!r} from {archive_path}.")
        try:
            tar.extractall(destination, filter="data")
        except TypeError:
            tar.extractall(destination)


def restore_blob(blob, output_dir: Path, restore_dir_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_dir = output_dir / restore_dir_name
    tmp_root = output_dir / ".gcs_restore_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=tmp_root, suffix=".tar", delete=False) as handle:
        archive_path = Path(handle.name)

    try:
        blob.download_to_filename(str(archive_path))
        extract_root = tmp_root / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        safe_extract_tar(archive_path, extract_root)

        extracted_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
        if len(extracted_dirs) != 1:
            raise RuntimeError(
                f"Expected exactly one top-level checkpoint directory in {blob.name}, found {len(extracted_dirs)}."
            )
        extracted_dir = extracted_dirs[0]
        trainer_path = extracted_dir / "trainer.pt"
        if not trainer_path.exists():
            raise RuntimeError(f"Restored checkpoint from {blob.name} is missing trainer.pt.")

        if restore_dir.exists():
            shutil.rmtree(restore_dir)
        shutil.move(str(extracted_dir), str(restore_dir))
        metadata = {
            "source_blob": blob.name,
            "bucket": blob.bucket.name,
            "restored_at_unix": int(time.time()),
        }
        (restore_dir / "restored_from_gcs.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return restore_dir / "trainer.pt"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    args = parse_args()
    if not args.bucket:
        raise SystemExit("Missing bucket target. Set --bucket or GCS_BUCKET.")

    prefix = (args.object_prefix or default_object_prefix()).strip("/")
    client = new_storage_client()
    bucket = client.bucket(args.bucket)
    blob, step = find_latest_step_blob(bucket, prefix)
    if blob is None:
        print(json.dumps({"status": "not_found", "bucket": args.bucket, "prefix": prefix}), flush=True)
        return

    trainer_path = restore_blob(blob, args.output_dir.resolve(), args.restore_dir_name)
    print(
        json.dumps(
            {
                "status": "restored",
                "bucket": args.bucket,
                "source_blob": blob.name,
                "step": step,
                "trainer_path": str(trainer_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
