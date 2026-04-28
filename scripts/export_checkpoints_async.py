#!/usr/bin/env python

import argparse
import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Asynchronously archive and export milestone checkpoints to Google Cloud Storage.")
    parser.add_argument("output_dir", type=Path, help="Training output directory containing step_* checkpoints.")
    parser.add_argument(
        "--poll-secs",
        type=float,
        default=float(os.environ.get("GCS_EXPORT_POLL_SECS", "15")),
        help="Polling interval for new checkpoints.",
    )
    parser.add_argument(
        "--run-prefix",
        default=os.environ.get("GCS_RUN_PREFIX"),
        help="Object prefix name for this run. Defaults to the output directory basename.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Local state directory. Defaults to OUTPUT_DIR/.gcs_export.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        default=os.environ.get("GCS_KEEP_ARCHIVES", "0") == "1",
        help="Keep local .tar archives after successful upload.",
    )
    return parser.parse_args()


class Exporter:
    def __init__(self, args):
        self.output_dir = args.output_dir.resolve()
        self.poll_secs = max(args.poll_secs, 1.0)
        self.run_prefix = args.run_prefix or self.output_dir.name
        self.base_prefix = os.environ.get("GCS_PREFIX", "")
        self.state_dir = (args.state_dir or (self.output_dir / ".gcs_export")).resolve()
        self.keep_archives = args.keep_archives
        self.stop_requested = False
        self.failure_counts: dict[str, int] = {}
        self.max_final_retries = int(os.environ.get("GCS_EXPORT_MAX_FINAL_RETRIES", "3"))

        self.stop_file = Path(os.environ.get("GCS_STOP_FILE", str(self.state_dir / "stop")))
        self.uploaded_dir = self.state_dir / "uploaded"
        self.archive_dir = self.state_dir / "archives"
        self.upload_state_dir = self.state_dir / "upload_state"
        self.uploader_script = Path(__file__).with_name("gcs_resumable_upload.py")
        self.gcs_bucket = os.environ.get("GCS_BUCKET")
        if not self.gcs_bucket:
            raise SystemExit("Missing GCS_BUCKET for checkpoint export.")

    def install_signal_handlers(self):
        loop = asyncio.get_running_loop()

        def request_stop():
            self.stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except NotImplementedError:  # pragma: no cover - not expected on Linux clusters
                signal.signal(sig, lambda *_: request_stop())

    async def run(self):
        self.install_signal_handlers()
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.upload_state_dir.mkdir(parents=True, exist_ok=True)

        print(
            json.dumps(
                {
                    "event": "exporter_started",
                    "output_dir": str(self.output_dir),
                    "run_prefix": self.run_prefix,
                    "poll_secs": self.poll_secs,
                }
            ),
            flush=True,
        )

        while True:
            pending = self.pending_checkpoints()
            if pending:
                checkpoint_dir = pending[0]
                try:
                    await self.process_checkpoint(checkpoint_dir)
                    self.failure_counts.pop(checkpoint_dir.name, None)
                except Exception as exc:  # pragma: no cover - runtime fault handling
                    self.failure_counts[checkpoint_dir.name] = self.failure_counts.get(checkpoint_dir.name, 0) + 1
                    print(
                        json.dumps(
                            {
                                "event": "export_failed",
                                "checkpoint": checkpoint_dir.name,
                                "attempt": self.failure_counts[checkpoint_dir.name],
                                "error": str(exc),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    if (self.stop_requested or self.stop_file.exists()) and self.failure_counts[checkpoint_dir.name] >= self.max_final_retries:
                        raise
                    await asyncio.sleep(self.poll_secs)
                continue

            if self.stop_requested or self.stop_file.exists():
                print(json.dumps({"event": "exporter_stopped"}), flush=True)
                return

            await asyncio.sleep(self.poll_secs)

    def pending_checkpoints(self) -> list[Path]:
        candidates: list[Path] = []
        for checkpoint_dir in sorted(self.output_dir.glob("step_*")):
            if not checkpoint_dir.is_dir():
                continue
            manifest_path = checkpoint_dir / "checkpoint_manifest.json"
            trainer_path = checkpoint_dir / "trainer.pt"
            done_path = self.uploaded_dir / f"{checkpoint_dir.name}.done.json"
            if manifest_path.exists() and trainer_path.exists() and not done_path.exists():
                candidates.append(checkpoint_dir)
        return candidates

    async def process_checkpoint(self, checkpoint_dir: Path):
        archive_path = self.archive_dir / f"{checkpoint_dir.name}.tar"
        done_path = self.uploaded_dir / f"{checkpoint_dir.name}.done.json"
        if done_path.exists():
            return

        if not archive_path.exists():
            await self.create_archive(checkpoint_dir, archive_path)

        upload_result = await self.upload_archive(archive_path)
        done_payload = {
            "checkpoint": checkpoint_dir.name,
            "archive": archive_path.name,
            "uploaded": upload_result,
        }
        done_path.write_text(json.dumps(done_payload, indent=2, sort_keys=True))
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        if not self.keep_archives and archive_path.exists():
            archive_path.unlink()

        print(
            json.dumps(
                {
                    "event": "checkpoint_exported",
                    "checkpoint": checkpoint_dir.name,
                    "archive": done_payload["archive"],
                }
            ),
            flush=True,
        )

    async def create_archive(self, checkpoint_dir: Path, archive_path: Path):
        tmp_path = archive_path.with_suffix(".tar.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        command = self.low_priority_command(
            [
                "tar",
                "-C",
                str(self.output_dir),
                "-cf",
                str(tmp_path),
                checkpoint_dir.name,
            ]
        )
        await self.run_subprocess(command, action=f"archive {checkpoint_dir.name}")
        tmp_path.replace(archive_path)

    async def upload_archive(self, archive_path: Path) -> dict:
        object_prefix_parts = [self.base_prefix.strip("/"), self.run_prefix.strip("/"), "checkpoints"]
        object_prefix = "/".join(part for part in object_prefix_parts if part)
        command = self.low_priority_command(
            [
                sys.executable,
                str(self.uploader_script),
                "--local-path",
                str(archive_path),
                "--bucket",
                self.gcs_bucket,
                "--object-prefix",
                object_prefix,
                "--state-dir",
                str(self.upload_state_dir),
            ]
        )
        stdout = await self.run_subprocess(command, action=f"upload {archive_path.name}", capture_stdout=True)
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"uploader produced no JSON output for {archive_path.name}")
        return json.loads(lines[-1])

    async def run_subprocess(self, command: list[str], *, action: str, capture_stdout: bool = False) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE if capture_stdout else None,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        if process.returncode != 0:
            raise RuntimeError(
                f"{action} failed with exit code {process.returncode}: {stderr_text.strip() or stdout_text.strip()}"
            )
        if stderr_text.strip():
            print(stderr_text.strip(), file=sys.stderr, flush=True)
        return stdout_text

    @staticmethod
    def low_priority_command(base_command: list[str]) -> list[str]:
        command = list(base_command)
        if shutil.which("nice"):
            command = ["nice", "-n", "10", *command]
        if shutil.which("ionice"):
            command = ["ionice", "-c3", *command]
        return command


async def async_main():
    args = parse_args()
    exporter = Exporter(args)
    await exporter.run()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
