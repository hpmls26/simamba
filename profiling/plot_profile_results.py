import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def to_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def plot_kernel_latency(rows, out_dir):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped[(row["kernel"], row["batch"])].append(row)
    plt.figure(figsize=(8, 5))
    for (kernel, batch), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["seqlen"]))
        plt.plot(
            [int(item["seqlen"]) for item in items],
            [to_float(item, "forward_ms_mean") for item in items],
            marker="o",
            label=f"{kernel}, batch={batch}",
        )
    plt.xlabel("Sequence length")
    plt.ylabel("Forward latency (ms)")
    plt.title("Kernel Forward Latency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kernel_latency.png", dpi=200)
    plt.close()


def plot_kernel_memory(rows, out_dir):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped[(row["kernel"], row["batch"])].append(row)
    plt.figure(figsize=(8, 5))
    for (kernel, batch), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["seqlen"]))
        plt.plot(
            [int(item["seqlen"]) for item in items],
            [to_float(item, "peak_memory_bytes") / (1024 ** 2) for item in items],
            marker="o",
            label=f"{kernel}, batch={batch}",
        )
    plt.xlabel("Sequence length")
    plt.ylabel("Peak allocated memory (MiB)")
    plt.title("Kernel Memory Scaling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kernel_memory.png", dpi=200)
    plt.close()


def plot_vllm_throughput(rows, out_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["max_tokens"], row["prompt_words"])].append(row)
    plt.figure(figsize=(9, 5))
    for (model, max_tokens, prompt_words), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["batch_size"]))
        label = f"{model.split('/')[-1]}, gen={max_tokens}, prompt_words={prompt_words}"
        plt.plot(
            [int(item["batch_size"]) for item in items],
            [to_float(item, "tokens_per_sec") for item in items],
            marker="o",
            label=label,
        )
    plt.xlabel("Batch size")
    plt.ylabel("Generated tokens/sec")
    plt.title("vLLM Throughput")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "vllm_throughput.png", dpi=200)
    plt.close()


def plot_vllm_latency(rows, out_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["max_tokens"])].append(row)
    plt.figure(figsize=(9, 5))
    for (model, max_tokens), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: (int(item["prompt_words"]), int(item["batch_size"])))
        x_labels = [f"p{item['prompt_words']}/b{item['batch_size']}" for item in items]
        plt.plot(x_labels, [to_float(item, "latency_sec") for item in items], marker="o", label=f"{model.split('/')[-1]}, gen={max_tokens}")
    plt.xlabel("Prompt words / batch size")
    plt.ylabel("Latency (sec)")
    plt.title("vLLM Latency")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "vllm_latency.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create report plots from profiling CSVs.")
    parser.add_argument("--kernel-csv", default="results/kernel_sweep.csv")
    parser.add_argument("--vllm-csv", default="results/vllm_sweep.csv")
    parser.add_argument("--out-dir", default="results/plots")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", default="profile_plots")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    kernel_path = root / args.kernel_csv
    if kernel_path.exists():
        kernel_rows = read_csv(kernel_path)
        plot_kernel_latency(kernel_rows, out_dir)
        plot_kernel_memory(kernel_rows, out_dir)
        generated.extend([out_dir / "kernel_latency.png", out_dir / "kernel_memory.png"])

    vllm_path = root / args.vllm_csv
    if vllm_path.exists():
        vllm_rows = read_csv(vllm_path)
        plot_vllm_throughput(vllm_rows, out_dir)
        plot_vllm_latency(vllm_rows, out_dir)
        generated.extend([out_dir / "vllm_throughput.png", out_dir / "vllm_latency.png"])

    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="profile_plots",
            name=args.wandb_name,
        )
        for path in generated:
            if path.exists():
                wandb.log({path.stem: wandb.Image(str(path))})
        artifact = wandb.Artifact("profile_plots", type="profile_plots")
        for path in generated:
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
