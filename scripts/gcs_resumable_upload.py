#!/usr/bin/env python

import argparse
import hashlib
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests


DEFAULT_CHUNK_MIB = 64
CHUNK_ALIGNMENT = 256 * 1024
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
INVALID_SESSION_STATUSES = {404, 410}


class GCSUploadError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable Google Cloud Storage uploader for large checkpoint archives.")
    parser.add_argument("--local-path", type=Path, required=True, help="Local file to upload.")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET"), help="Target GCS bucket name.")
    parser.add_argument(
        "--object-prefix",
        default=os.environ.get("GCS_PREFIX", ""),
        help="Optional object prefix for the uploaded file.",
    )
    parser.add_argument("--remote-name", default=None, help="Remote object name. Defaults to the local filename.")
    parser.add_argument(
        "--content-type",
        default=None,
        help="Explicit content type. Defaults to a mimetype guess or application/octet-stream.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory used to store resumable upload resume state.",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=int(os.environ.get("GCS_UPLOAD_CHUNK_MIB", DEFAULT_CHUNK_MIB)),
        help="Chunk size in MiB for resumable uploads.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("GCS_UPLOAD_MAX_RETRIES", "8")),
        help="Maximum session recreation retries before failing.",
    )
    return parser.parse_args()


def chunk_size_bytes(chunk_size_mib: int) -> int:
    if chunk_size_mib <= 0:
        raise ValueError("--chunk-size-mib must be positive.")
    size = chunk_size_mib * 1024 * 1024
    return max(CHUNK_ALIGNMENT, (size // CHUNK_ALIGNMENT) * CHUNK_ALIGNMENT)


def build_object_name(prefix: str, remote_name: str) -> str:
    parts = [segment.strip("/") for segment in [prefix, remote_name] if segment and segment.strip("/")]
    if not parts:
        raise ValueError("Object name cannot be empty.")
    return "/".join(parts)


def compute_state_path(state_dir: Path, local_path: Path, bucket: str, object_name: str) -> Path:
    digest = hashlib.sha1(f"{local_path.resolve()}::{bucket}::{object_name}".encode("utf-8")).hexdigest()
    return state_dir / f"{digest}.json"


def load_state(state_path: Path, local_path: Path) -> Optional[dict[str, Any]]:
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    stat = local_path.stat()
    if state.get("local_size") != stat.st_size or state.get("local_mtime_ns") != stat.st_mtime_ns:
        return None
    return state


def save_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(state_path)


def delete_state(state_path: Path) -> None:
    if state_path.exists():
        state_path.unlink()


def new_storage_client():
    try:
        from google.cloud import storage
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime envs
        raise SystemExit(
            "GCS export requires 'google-cloud-storage'. "
            f"Install the repo train extras with: {sys.executable} -m pip install -e '.[train]'"
        ) from exc

    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return storage.Client(project=project)
    return storage.Client()


def get_existing_blob(client, bucket_name: str, object_name: str):
    bucket = client.bucket(bucket_name)
    return bucket.get_blob(object_name)


def create_resumable_session(client, bucket_name: str, object_name: str, *, content_type: str, total_size: int) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    session_uri = blob.create_resumable_upload_session(
        content_type=content_type,
        size=total_size,
        client=client,
    )
    if not session_uri:
        raise GCSUploadError(f"GCS did not return a resumable session URI for {object_name}.")
    return session_uri


def parse_uploaded_bytes(range_header: Optional[str]) -> int:
    if not range_header:
        return 0
    try:
        _, byte_range = range_header.split("=")
        _, end_str = byte_range.split("-")
        return int(end_str) + 1
    except (ValueError, AttributeError) as exc:
        raise GCSUploadError(f"Unexpected Range header from GCS: {range_header!r}") from exc


def maybe_parse_json(response: requests.Response) -> Optional[dict[str, Any]]:
    try:
        return response.json()
    except ValueError:
        return None


def format_http_error(response: requests.Response) -> str:
    payload = maybe_parse_json(response)
    if payload is None:
        payload = response.text.strip()
    return f"HTTP {response.status_code}: {payload}"


def query_upload_offset(session_uri: str, total_size: int) -> Optional[int]:
    response = requests.put(
        session_uri,
        headers={"Content-Length": "0", "Content-Range": f"bytes */{total_size}"},
        timeout=(30, 300),
    )
    if response.status_code in (200, 201):
        return total_size
    if response.status_code == 308:
        return parse_uploaded_bytes(response.headers.get("Range"))
    if response.status_code in INVALID_SESSION_STATUSES:
        return None
    if response.status_code in RETRYABLE_STATUSES:
        raise GCSUploadError(f"GCS returned {response.status_code} while checking upload progress.")
    raise GCSUploadError(f"Unable to query resumable upload progress: {format_http_error(response)}")


def upload_chunks(
    *,
    local_path: Path,
    session_uri: str,
    total_size: int,
    chunk_size: int,
    starting_offset: int,
    state_path: Path,
    state_payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    offset = starting_offset
    with local_path.open("rb") as handle:
        handle.seek(offset)
        while offset < total_size:
            remaining = total_size - offset
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise GCSUploadError(f"Unexpected EOF while reading {local_path}.")
            end_offset = offset + len(chunk) - 1
            headers = {
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {offset}-{end_offset}/{total_size}",
            }

            try:
                response = requests.put(session_uri, headers=headers, data=chunk, timeout=(30, 1800))
            except requests.RequestException as exc:
                print(f"chunk upload error at byte {offset}: {exc}", file=sys.stderr, flush=True)
                progressed = query_upload_offset(session_uri, total_size)
                if progressed is None:
                    return None
                offset = progressed
                handle.seek(offset)
                state_payload["uploaded_bytes"] = offset
                save_state(state_path, state_payload)
                continue

            if response.status_code in (200, 201):
                return maybe_parse_json(response) or {}
            if response.status_code == 308:
                offset = parse_uploaded_bytes(response.headers.get("Range"))
                handle.seek(offset)
                state_payload["uploaded_bytes"] = offset
                save_state(state_path, state_payload)
                continue
            if response.status_code in INVALID_SESSION_STATUSES:
                return None
            if response.status_code in RETRYABLE_STATUSES:
                progressed = query_upload_offset(session_uri, total_size)
                if progressed is None:
                    return None
                offset = progressed
                handle.seek(offset)
                state_payload["uploaded_bytes"] = offset
                save_state(state_path, state_payload)
                continue
            raise GCSUploadError(f"Chunk upload failed for {local_path.name}: {format_http_error(response)}")
    return None


def upload_file(args) -> dict[str, Any]:
    local_path = args.local_path.resolve()
    if not local_path.is_file():
        raise SystemExit(f"Local upload target does not exist: {local_path}")
    if not args.bucket:
        raise SystemExit("Missing bucket target. Set --bucket or GCS_BUCKET.")

    remote_name = args.remote_name or local_path.name
    object_name = build_object_name(args.object_prefix, remote_name)
    content_type = args.content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    chunk_size = chunk_size_bytes(args.chunk_size_mib)
    total_size = local_path.stat().st_size

    state_dir = args.state_dir or (local_path.parent / ".gcs_upload_state")
    state_path = compute_state_path(state_dir, local_path, args.bucket, object_name)
    state = load_state(state_path, local_path)
    if state is None and state_path.exists():
        delete_state(state_path)

    client = new_storage_client()
    existing = get_existing_blob(client, args.bucket, object_name)
    if existing is not None and int(existing.size or -1) == total_size:
        delete_state(state_path)
        return {
            "status": "skipped_existing",
            "bucket": args.bucket,
            "object_name": object_name,
            "size": total_size,
            "generation": getattr(existing, "generation", None),
        }

    session_uri = state.get("session_uri") if state is not None else None
    uploaded_bytes = int(state.get("uploaded_bytes", 0)) if state is not None else 0
    base_state = {
        "local_path": str(local_path),
        "local_size": total_size,
        "local_mtime_ns": local_path.stat().st_mtime_ns,
        "bucket": args.bucket,
        "object_name": object_name,
        "content_type": content_type,
        "chunk_size": chunk_size,
        "session_uri": session_uri,
        "uploaded_bytes": uploaded_bytes,
    }

    for attempt in range(1, args.max_retries + 1):
        if not session_uri:
            try:
                session_uri = create_resumable_session(
                    client,
                    args.bucket,
                    object_name,
                    content_type=content_type,
                    total_size=total_size,
                )
            except Exception as exc:
                if attempt == args.max_retries:
                    raise
                sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random())
                print(
                    f"retrying resumable session creation for {object_name} after error: {exc} (sleep {sleep_s:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            base_state["session_uri"] = session_uri
            base_state["uploaded_bytes"] = uploaded_bytes
            save_state(state_path, base_state)

        offset = query_upload_offset(session_uri, total_size)
        if offset is None:
            existing = get_existing_blob(client, args.bucket, object_name)
            if existing is not None and int(existing.size or -1) == total_size:
                delete_state(state_path)
                return {
                    "status": "uploaded",
                    "bucket": args.bucket,
                    "object_name": object_name,
                    "size": total_size,
                    "generation": getattr(existing, "generation", None),
                }
            session_uri = None
            base_state["session_uri"] = None
            base_state["uploaded_bytes"] = 0
            save_state(state_path, base_state)
            continue
        if offset == total_size:
            delete_state(state_path)
            existing = get_existing_blob(client, args.bucket, object_name)
            return {
                "status": "uploaded",
                "bucket": args.bucket,
                "object_name": object_name,
                "size": total_size,
                "generation": getattr(existing, "generation", None) if existing is not None else None,
            }

        base_state["uploaded_bytes"] = offset
        save_state(state_path, base_state)
        result = upload_chunks(
            local_path=local_path,
            session_uri=session_uri,
            total_size=total_size,
            chunk_size=chunk_size,
            starting_offset=offset,
            state_path=state_path,
            state_payload=base_state,
        )
        if result is not None:
            delete_state(state_path)
            return {
                "status": "uploaded",
                "bucket": args.bucket,
                "object_name": object_name,
                "size": total_size,
                "generation": result.get("generation"),
            }

        existing = get_existing_blob(client, args.bucket, object_name)
        if existing is not None and int(existing.size or -1) == total_size:
            delete_state(state_path)
            return {
                "status": "uploaded",
                "bucket": args.bucket,
                "object_name": object_name,
                "size": total_size,
                "generation": getattr(existing, "generation", None),
            }

        session_uri = None
        base_state["session_uri"] = None
        base_state["uploaded_bytes"] = 0
        save_state(state_path, base_state)

    raise GCSUploadError(f"Failed to upload {local_path} after {args.max_retries} resumable session attempts.")


def main():
    args = parse_args()
    result = upload_file(args)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
