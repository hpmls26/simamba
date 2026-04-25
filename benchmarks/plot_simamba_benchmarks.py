#!/usr/bin/env python3

"""Plot Simamba vs Mamba-3 benchmark results using matplotlib.

This script encodes benchmark values collected from terminal runs and writes
PNG figures for latency, throughput, memory, and relative ratio views.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MetricPair = Tuple[float, float]  # (mamba3, simamba)


RUNS: List[Dict[str, object]] = [
    {
        "label": "B1 P128 G64",
        "metrics": {
            "prefill_ms": (0.0522, 40.5555),
            "prefill_toks_s": (2450980.4366, 3156.1672),
            "step_ms": (0.0063, 0.0068),
            "step_toks_s": (158569.3399, 147222.6783),
            "prefill_peak_mb": (5.8833, 5.9043),
            "step_peak_mb": (5.3887, 5.3965),
        },
    },
    {
        "label": "B1 P1024 G256",
        "metrics": {
            "e2e_ms": (64.1028, 403.5323),
            "e2e_total_toks_s": (19967.9238, 3171.9892),
            "e2e_gen_toks_s": (3993.5848, 634.3978),
        },
    },
    {
        "label": "B4 P1024 G256",
        "metrics": {
            "prefill_ms": (0.8602, 353.3066),
            "prefill_toks_s": (4761904.7997, 11593.3297),
            "step_ms": (0.0160, 0.0173),
            "step_toks_s": (249673.9159, 231759.5575),
            "prefill_peak_mb": (167.3911, 124.2363),
            "step_peak_mb": (108.1738, 108.2051),
            "e2e_ms": (73.3674, 442.9353),
            "e2e_total_toks_s": (69785.7282, 11559.2501),
            "e2e_gen_toks_s": (13957.1456, 2311.8500),
        },
    },
    {
        "label": "B4 P1024 G256 Mid",
        "metrics": {
            "prefill_ms": (0.8602, 365.0673),
            "prefill_toks_s": (4761904.7997, 11219.8503),
            "step_ms": (0.0160, 0.0180),
            "step_toks_s": (249457.8239, 222593.4728),
            "prefill_peak_mb": (168.0161, 124.8618),
            "step_peak_mb": (108.7988, 108.8306),
            "e2e_ms": (72.4556, 465.6271),
            "e2e_total_toks_s": (70663.9217, 10995.9232),
            "e2e_gen_toks_s": (14132.7843, 2199.1846),
        },
    },
]


METRIC_LABELS = {
    "prefill_ms": "Prefill ms",
    "step_ms": "Step ms",
    "e2e_ms": "E2E ms",
    "prefill_toks_s": "Prefill tok/s",
    "step_toks_s": "Step tok/s",
    "e2e_total_toks_s": "E2E total tok/s",
    "e2e_gen_toks_s": "E2E gen tok/s",
    "prefill_peak_mb": "Prefill peak MB",
    "step_peak_mb": "Step peak MB",
}

LATENCY_METRICS = ["prefill_ms", "step_ms", "e2e_ms"]
THROUGHPUT_METRICS = ["prefill_toks_s", "step_toks_s", "e2e_total_toks_s", "e2e_gen_toks_s"]
MEMORY_METRICS = ["prefill_peak_mb", "step_peak_mb"]


def _collect_points(metric_names: List[str]) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    for run in RUNS:
        run_label = str(run["label"])
        metrics = run["metrics"]
        assert isinstance(metrics, dict)
        for metric_name in metric_names:
            if metric_name in metrics:
                m3, sim = metrics[metric_name]
                points.append((f"{run_label}\n{METRIC_LABELS[metric_name]}", float(m3), float(sim)))
    return points


def _plot_grouped_bars(
    points: List[Tuple[str, float, float]],
    *,
    title: str,
    y_label: str,
    out_path: Path,
    log_scale: bool = True,
) -> None:
    x_labels = [p[0] for p in points]
    m3_vals = np.array([p[1] for p in points], dtype=np.float64)
    sim_vals = np.array([p[2] for p in points], dtype=np.float64)
    ratios = sim_vals / np.maximum(m3_vals, 1e-12)

    x = np.arange(len(points))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(points) * 1.2), 6.5), constrained_layout=True)
    bars_m3 = ax.bar(x - width / 2, m3_vals, width, label="Mamba-3", color="#4C78A8")
    bars_sim = ax.bar(x + width / 2, sim_vals, width, label="Simamba", color="#F58518")

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    for idx, bar in enumerate(bars_sim):
        y = bar.get_height()
        if y <= 0:
            continue
        ratio_txt = f"{ratios[idx]:.2f}x"
        y_text = y * (1.12 if log_scale else 1.02)
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, ratio_txt, ha="center", va="bottom", fontsize=8)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ratio_summary(out_path: Path) -> None:
    groups = [
        ("Latency: sim/m3 (lower is better)", LATENCY_METRICS),
        ("Throughput: sim/m3 (higher is better)", THROUGHPUT_METRICS),
        ("Memory: sim/m3", MEMORY_METRICS),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 13), constrained_layout=True)

    for ax, (title, metrics) in zip(axes, groups):
        points = _collect_points(metrics)
        labels = [p[0] for p in points]
        ratios = np.array([p[2] / max(p[1], 1e-12) for p in points], dtype=np.float64)

        x = np.arange(len(points))
        colors = ["#E45756" if r > 1.0 else "#54A24B" for r in ratios]
        ax.bar(x, ratios, color=colors)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("sim/m3")
        ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Simamba benchmark results with matplotlib.")
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/plots"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    latency_points = _collect_points(LATENCY_METRICS)
    throughput_points = _collect_points(THROUGHPUT_METRICS)
    memory_points = _collect_points(MEMORY_METRICS)

    latency_out = outdir / "simamba_latency.png"
    throughput_out = outdir / "simamba_throughput.png"
    memory_out = outdir / "simamba_memory.png"
    ratio_out = outdir / "simamba_ratio_summary.png"

    _plot_grouped_bars(
        latency_points,
        title="Simamba vs Mamba-3 Latency",
        y_label="Milliseconds (log scale)",
        out_path=latency_out,
        log_scale=True,
    )
    _plot_grouped_bars(
        throughput_points,
        title="Simamba vs Mamba-3 Throughput",
        y_label="Tokens/second (log scale)",
        out_path=throughput_out,
        log_scale=True,
    )
    _plot_grouped_bars(
        memory_points,
        title="Simamba vs Mamba-3 Peak Memory",
        y_label="MB (log scale)",
        out_path=memory_out,
        log_scale=True,
    )
    _plot_ratio_summary(ratio_out)

    print("Wrote graphs:")
    print(f"- {latency_out}")
    print(f"- {throughput_out}")
    print(f"- {memory_out}")
    print(f"- {ratio_out}")


if __name__ == "__main__":
    main()
