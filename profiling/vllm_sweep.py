import argparse
import csv
import math
import os
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from vllm import LLM, SamplingParams

from vllm_profiler import mamba2_hf_overrides


GROUP_KEYS = (
    "model",
    "prefix_cache",
    "prompt_kind",
    "batch_size",
    "prompt_words",
    "repeated_prefix_tokens",
    "max_tokens",
)

SUMMARY_METRICS = (
    "latency_sec",
    "ttft_sec",
    "tpot_sec",
    "tokens_per_sec",
    "requests_per_sec",
    "decode_loop_sec",
    "decode_tokens_per_sec",
    "prefill_probe_latency_sec",
    "prefill_probe_ttft_sec",
    "gpu_memory_peak_used_bytes",
    "gpu_memory_peak_delta_bytes",
    "model_memory_delta_load_bytes",
)


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def make_prompt(target_words):
    base = "State space sequence models are efficient for long context generation because "
    words = base.split()
    repeats = (target_words + len(words) - 1) // len(words)
    return " ".join((words * repeats)[:target_words])


def parse_modes(value):
    modes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in {"on", "off"}:
            raise ValueError(f"Expected prefix cache mode on/off, got {item!r}")
        modes.append(item)
    return modes


def make_repeated_prefix_token_prompts(tokenizer, prefix_tokens, batch_size):
    prefix_text = " ".join(["prefix"] * max(1, prefix_tokens * 2))
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)[:prefix_tokens]
    if len(prefix_ids) < prefix_tokens:
        raise RuntimeError(f"Could only build {len(prefix_ids)} prefix tokens, requested {prefix_tokens}")
    prompts = []
    for idx in range(batch_size):
        suffix_ids = tokenizer.encode(f" request {idx}", add_special_tokens=False)
        prompts.append({"prompt_token_ids": prefix_ids + suffix_ids})
    return prompts, len(prefix_ids)


def request_metric(output, name):
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    value = getattr(metrics, name, None)
    return float(value) if value is not None else None


def summarize_request_metrics(outputs, elapsed):
    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    ttfts = [request_metric(output, "first_token_latency") for output in outputs]
    ttfts = [value for value in ttfts if value is not None and value > 0]
    decode_times = []
    for output in outputs:
        first = request_metric(output, "first_token_ts")
        last = request_metric(output, "last_token_ts")
        if first is not None and last is not None and last >= first:
            decode_times.append(last - first)
    per_request_generated = [len(output.outputs[0].token_ids) for output in outputs]
    decode_tokens = sum(max(0, count - 1) for count in per_request_generated)
    decode_time_sum = sum(decode_times)
    if decode_time_sum > 0 and decode_tokens > 0:
        tpot = decode_time_sum / decode_tokens
    elif generated > len(outputs):
        tpot = max(0.0, elapsed - (sum(ttfts) / len(ttfts) if ttfts else 0.0)) / (generated - len(outputs))
    else:
        tpot = 0.0
    return {
        "generated_tokens": generated,
        "ttft_sec": sum(ttfts) / len(ttfts) if ttfts else "",
        "decode_loop_sec": decode_time_sum / len(decode_times) if decode_times else 0.0,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_sec": decode_tokens / decode_time_sum if decode_time_sum > 0 else 0.0,
        "tpot_sec": tpot,
        "tokens_per_sec": generated / elapsed,
        "requests_per_sec": len(outputs) / elapsed,
    }


def prefix_keys(prefix, values):
    return {f"{prefix}{key}": value for key, value in values.items()}


def gpu_memory_used_bytes(device):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used_mib, total_mib = [int(item.strip()) for item in output.split(",")[:2]]
        return used_mib * 1024 * 1024, total_mib * 1024 * 1024
    except Exception:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return int(total_bytes - free_bytes), int(total_bytes)


class GpuMemorySampler:
    def __init__(self, device=0, interval_sec=0.005):
        self.device = device
        self.interval_sec = interval_sec
        self.start_used = 0
        self.end_used = 0
        self.peak_used = 0
        self.total = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start_used, self.total = gpu_memory_used_bytes(self.device)
        self.end_used = self.start_used
        self.peak_used = self.start_used
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            try:
                used, total = gpu_memory_used_bytes(self.device)
                self.total = total
                self.peak_used = max(self.peak_used, used)
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def __exit__(self, exc_type, exc, tb):
        self.end_used, self.total = gpu_memory_used_bytes(self.device)
        self.peak_used = max(self.peak_used, self.end_used)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def as_dict(self, prefix="gpu_memory_"):
        return {
            f"{prefix}total_bytes": self.total,
            f"{prefix}start_used_bytes": self.start_used,
            f"{prefix}peak_used_bytes": self.peak_used,
            f"{prefix}end_used_bytes": self.end_used,
            f"{prefix}peak_delta_bytes": max(0, self.peak_used - self.start_used),
        }


def measure_generate(llm, prompts, sampling, device, memory_poll_interval):
    with GpuMemorySampler(device=device, interval_sec=memory_poll_interval) as memory:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
    metrics = summarize_request_metrics(outputs, elapsed)
    return outputs, elapsed, metrics, memory.as_dict()


def force_transformers_backend_compat(model):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import vllm.model_executor.models.registry as vllm_registry
    import vllm.model_executor.models.transformers.base as vllm_tf_base

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", {}) or {}
    target = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if target is None:
        return
    model_cls = get_class_from_dynamic_module(target, model, trust_remote_code=True)
    model_cls._supports_attention_backend = True
    model_cls.tp_plan = property(lambda self: getattr(self, "_tp_plan", None) or {})
    vllm_tf_base.Base.recursive_replace = lambda self: None

    original_try_get = vllm_registry.try_get_class_from_dynamic_module

    def patched_try_get(module, model_name, *args, **kwargs):
        cls = original_try_get(module, model_name, *args, **kwargs)
        if cls is not None and cls.__name__.startswith("Simamba"):
            cls._supports_attention_backend = True
            cls.tp_plan = property(lambda self: getattr(self, "_tp_plan", None) or {})
        return cls

    vllm_registry.try_get_class_from_dynamic_module = patched_try_get


def fieldnames_for(rows):
    names = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(rows))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, pct):
    if not values:
        return ""
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil((pct / 100.0) * len(values)) - 1))
    return values[idx]


def numeric_stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "mean": "",
            "median": "",
            "p95": "",
            "std": "",
            "min": "",
            "max": "",
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_samples(rows):
    grouped = defaultdict(list)
    failed = []
    for row in rows:
        if row.get("status") == "ok":
            grouped[tuple(row.get(key, "") for key in GROUP_KEYS)].append(row)
        else:
            failed.append(row)

    summary_rows = []
    for key, items in sorted(grouped.items()):
        summary = {group_key: key[idx] for idx, group_key in enumerate(GROUP_KEYS)}
        summary.update({
            "timestamp": int(time.time()),
            "status": "ok",
            "sample_count": len(items),
        })
        for metric in SUMMARY_METRICS:
            stats = numeric_stats([as_float(item.get(metric)) for item in items])
            for stat_name, stat_value in stats.items():
                summary[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(summary)

    for row in failed:
        summary_rows.append({
            "timestamp": int(time.time()),
            "status": row.get("status", "failed"),
            "sample_count": 0,
            "model": row.get("model", ""),
            "prefix_cache": row.get("prefix_cache", ""),
            "prompt_kind": row.get("prompt_kind", ""),
            "batch_size": row.get("batch_size", ""),
            "prompt_words": row.get("prompt_words", ""),
            "repeated_prefix_tokens": row.get("repeated_prefix_tokens", ""),
            "max_tokens": row.get("max_tokens", ""),
            "error": row.get("error", ""),
        })
    return summary_rows


def short_model_name(model):
    return model.split("/")[-1]


def plot_summary(summary_rows, out_path):
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if not ok_rows:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    labels = [
        f"{short_model_name(row['model'])}\ncache={row['prefix_cache']}\n{row['prompt_kind']}"
        for row in ok_rows
    ]
    plots = [
        ("ttft_sec_median", "Median TTFT (ms)", 1000.0),
        ("tpot_sec_median", "Median TPOT (ms)", 1000.0),
        ("tokens_per_sec_median", "Median tok/s", 1.0),
        ("gpu_memory_peak_used_bytes_median", "Median GPU peak used (MiB)", 1.0 / (1024 ** 2)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (metric, title, scale) in zip(axes, plots):
        values = [(as_float(row.get(metric)) or 0.0) * scale for row in ok_rows]
        ax.bar(range(len(values)), values, color="#3b82f6")
        ax.set_title(title)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("vLLM Repeated Profile Summary", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def wandb_table(rows):
    if not rows:
        return None
    columns = fieldnames_for(rows)
    string_columns = {
        "model",
        "prefix_cache",
        "prompt_kind",
        "prompt_words",
        "repeated_prefix_tokens",
        "status",
        "error",
    }
    data = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if column in string_columns:
                values.append("" if value is None else str(value))
            else:
                values.append(value)
        data.append(values)
    return wandb.Table(columns=columns, data=data)


def output_paths(root, out_arg, raw_out_arg, plot_out_arg):
    summary_path = root / out_arg
    if raw_out_arg:
        raw_path = root / raw_out_arg
    else:
        raw_path = summary_path.with_name(f"{summary_path.stem}_raw.csv")
    if plot_out_arg:
        plot_path = root / plot_out_arg
    else:
        plot_path = summary_path.with_name(f"{summary_path.stem}_summary.png")
    return summary_path, raw_path, plot_path


def main():
    parser = argparse.ArgumentParser(description="Sweep vLLM throughput by model, prompt length, and batch size.")
    parser.add_argument(
        "--models",
        default="soumil1/mamba2-10m-slimpajama-500m,soumil1/simamba-midpoint-10m-slimpajama-500m",
    )
    parser.add_argument("--prompt-words", default="32,128,512")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--max-tokens", default="32,128")
    parser.add_argument("--prefix-cache-modes", default="off,on")
    parser.add_argument("--repeated-prefix-tokens", type=int, default=512)
    parser.add_argument("--mamba-block-size", type=int, default=16)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--force-transformers-backend-compatible", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-probe-tokens", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--memory-poll-interval-sec", type=float, default=0.005)
    parser.add_argument("--out", default="results/vllm_sweep.csv")
    parser.add_argument("--raw-out", default="")
    parser.add_argument("--plot-out", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="vllm_sweep")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    raw_rows = []
    root = Path(__file__).resolve().parent
    summary_path, raw_path, plot_path = output_paths(root, args.out, args.raw_out, args.plot_out)
    run = None
    if args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "profiling"),
            job_type="vllm_sweep",
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )

    for model in [item for item in args.models.split(",") if item]:
        if args.force_transformers_backend_compatible:
            force_transformers_backend_compat(model)
        for prefix_mode in parse_modes(args.prefix_cache_modes):
            cache_kwargs = {}
            if prefix_mode == "on":
                cache_kwargs["mamba_block_size"] = args.mamba_block_size
            load_before, gpu_total = gpu_memory_used_bytes(args.cuda_device)
            try:
                llm = LLM(
                    model=model,
                    tokenizer="EleutherAI/gpt-neox-20b",
                    tensor_parallel_size=args.tensor_parallel,
                    trust_remote_code=args.trust_remote_code,
                    model_impl=args.model_impl,
                    enforce_eager=True,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=max(parse_ints(args.batch_sizes)),
                    max_num_batched_tokens=args.max_model_len,
                    hf_overrides=mamba2_hf_overrides,
                    enable_prefix_caching=(prefix_mode == "on"),
                    disable_log_stats=False,
                    **cache_kwargs,
                )
            except Exception as exc:
                for batch_size in parse_ints(args.batch_sizes):
                    for max_tokens in parse_ints(args.max_tokens):
                        for prompt_kind in ("normal", "repeated_prefix"):
                            row = {
                                "timestamp": int(time.time()),
                                "model": model,
                                "prefix_cache": prefix_mode,
                                "prompt_kind": prompt_kind,
                                "batch_size": batch_size,
                                "prompt_words": parse_ints(args.prompt_words)[0] if prompt_kind == "normal" else "",
                                "repeated_prefix_tokens": args.repeated_prefix_tokens if prompt_kind == "repeated_prefix" else "",
                                "max_tokens": max_tokens,
                                "status": "load_failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            raw_rows.append(row)
                            write_csv(raw_path, raw_rows)
                            print(row, flush=True)
                continue

            tokenizer = llm.get_tokenizer()
            load_after, gpu_total = gpu_memory_used_bytes(args.cuda_device)
            load_memory = {
                "gpu_memory_total_bytes": gpu_total,
                "model_memory_before_load_bytes": load_before,
                "model_memory_after_load_bytes": load_after,
                "model_memory_delta_load_bytes": max(0, load_after - load_before),
            }

            for prompt_words in parse_ints(args.prompt_words):
                for batch_size in parse_ints(args.batch_sizes):
                    prompt_sets = [("normal", [make_prompt(prompt_words) for _ in range(batch_size)], "")]
                    repeated_prompts, actual_prefix_tokens = make_repeated_prefix_token_prompts(
                        tokenizer, args.repeated_prefix_tokens, batch_size
                    )
                    prompt_sets.append(("repeated_prefix", repeated_prompts, actual_prefix_tokens))
                    for prompt_kind, prompts, repeated_prefix_tokens in prompt_sets:
                        for max_tokens in parse_ints(args.max_tokens):
                            sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
                            prefill_sampling = SamplingParams(temperature=0.0, max_tokens=args.prefill_probe_tokens)
                            for _ in range(args.warmup):
                                llm.generate(prompts, prefill_sampling, use_tqdm=False)
                                llm.generate(prompts, sampling, use_tqdm=False)
                            for repeat in range(args.repeats):
                                _, prefill_elapsed, prefill_metrics, prefill_memory = measure_generate(
                                    llm,
                                    prompts,
                                    prefill_sampling,
                                    args.cuda_device,
                                    args.memory_poll_interval_sec,
                                )
                                _, elapsed, metrics, memory = measure_generate(
                                    llm,
                                    prompts,
                                    sampling,
                                    args.cuda_device,
                                    args.memory_poll_interval_sec,
                                )
                                row = {
                                    "timestamp": int(time.time()),
                                    "model": model,
                                    "prefix_cache": prefix_mode,
                                    "prompt_kind": prompt_kind,
                                    "batch_size": batch_size,
                                    "prompt_words": prompt_words if prompt_kind == "normal" else "",
                                    "repeated_prefix_tokens": repeated_prefix_tokens,
                                    "max_tokens": max_tokens,
                                    "repeat": repeat,
                                    "warmup": args.warmup,
                                    "repeats": args.repeats,
                                    "prefill_probe_tokens": args.prefill_probe_tokens,
                                    "latency_sec": elapsed,
                                    "status": "ok",
                                    "error": "",
                                }
                                row.update(load_memory)
                                row.update(memory)
                                row["peak_memory_bytes"] = memory["gpu_memory_peak_used_bytes"]
                                row.update(metrics)
                                row["prefill_probe_latency_sec"] = prefill_elapsed
                                row.update(prefix_keys("prefill_probe_", prefill_metrics))
                                row.update(prefix_keys("prefill_probe_", prefill_memory))
                                raw_rows.append(row)
                                summary_rows = summarize_samples(raw_rows)
                                write_csv(raw_path, raw_rows)
                                write_csv(summary_path, summary_rows)
                                print(row, flush=True)
                                if run is not None:
                                    wandb.log(row)
            del llm

    summary_rows = summarize_samples(raw_rows)
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)
    plot_summary(summary_rows, plot_path)
    if run is not None:
        if summary_rows:
            summary_table = wandb_table(summary_rows)
            if summary_table is not None:
                wandb.log({"vllm_summary_table": summary_table})
        if raw_rows:
            raw_table = wandb_table(raw_rows)
            if raw_table is not None:
                wandb.log({"vllm_raw_samples_table": raw_table})
        if plot_path.exists():
            wandb.log({"vllm_summary_plot": wandb.Image(str(plot_path))})
        artifact = wandb.Artifact("vllm_sweep_results", type="profile_results")
        artifact.add_file(str(summary_path))
        artifact.add_file(str(raw_path))
        if plot_path.exists():
            artifact.add_file(str(plot_path))
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
