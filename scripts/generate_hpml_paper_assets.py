#!/usr/bin/env python3
"""Generate figures and tables for docs/paper.tex from local run logs."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_LOGS = ROOT / "run_logs"
OUT = ROOT / "docs" / "assets" / "paper"


RUN_LABELS = {
    "outputs/disc10m_mamba2_fp32_500m_20260502_185217": "Mamba2",
    "outputs/disc10m_simamba_fp32_500m_20260502_190333": "Simamba Simpson",
    "outputs/disc10m_simamba_midpoint_fp32_500m_20260502_190333": "Simamba Simpson + midpoint",
    "outputs/disc10m_simamba_trapezoid_vec_500m_20260503_0609_trap_vec": "Simamba trapezoid",
    "outputs/disc10m_ablate_trapezoid_vec_50m_20260503_followup_50m": "Trapezoid",
    "outputs/disc10m_ablate_simpson_default_50m_20260503_followup_50m": "Simpson default",
    "outputs/disc10m_ablate_simpson_lowctrl_50m_20260503_followup_50m": "Simpson low-control",
    "outputs/disc10m_ablate_mamba2_dconv2_50m_20260503_dconv2_50m": "Mamba2 d_conv=2",
    "outputs/disc10m_ablate_mamba2_dconv1_nomem_50m_20260504": "Mamba2 d_conv=1",
    "outputs/disc10m_simamba_localconv_trapezoid_50m_20260504_localconv_dconv4_50m": "Local-conv trapezoid",
    "outputs/disc10m_simamba_localconv_simpson_lowctrl_50m_20260504_localconv_dconv4_50m": "Local-conv Simpson low-control",
    "outputs/disc10m_simamba_localconv_simpson_50m_20260504_localconv_dconv4_50m": "Local-conv Simpson",
    "outputs/simamba_2day_50m_seq128_100m_lr1e4_20260502_025645": "Unstable fp16/lr=1e-4 run",
}


MAIN_500M = [
    "outputs/disc10m_mamba2_fp32_500m_20260502_185217",
    "outputs/disc10m_simamba_fp32_500m_20260502_190333",
    "outputs/disc10m_simamba_midpoint_fp32_500m_20260502_190333",
    "outputs/disc10m_simamba_trapezoid_vec_500m_20260503_0609_trap_vec",
]
ABLATION_50M = [
    "outputs/disc10m_ablate_trapezoid_vec_50m_20260503_followup_50m",
    "outputs/disc10m_ablate_simpson_default_50m_20260503_followup_50m",
    "outputs/disc10m_ablate_simpson_lowctrl_50m_20260503_followup_50m",
]
LOCAL_MIX = [
    "outputs/disc10m_ablate_mamba2_dconv2_50m_20260503_dconv2_50m",
    "outputs/disc10m_ablate_mamba2_dconv1_nomem_50m_20260504",
    "outputs/disc10m_simamba_localconv_trapezoid_50m_20260504_localconv_dconv4_50m",
    "outputs/disc10m_simamba_localconv_simpson_lowctrl_50m_20260504_localconv_dconv4_50m",
    "outputs/disc10m_simamba_localconv_simpson_50m_20260504_localconv_dconv4_50m",
]


def parse_run_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    events: list[dict] = []
    for path in sorted(RUN_LOGS.glob("*.log")):
        current_output = None
        current_run_id = None
        current_run_name = None
        for raw in path.read_text(errors="ignore").splitlines():
            if not raw.startswith("{"):
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            if event == "startup.args_parsed":
                current_output = rec.get("output_dir", current_output)
                events.append({"log": str(path), **rec})
                continue
            if event == "wandb.init.end":
                current_run_id = rec.get("run_id", current_run_id)
                current_run_name = rec.get("run_name", current_run_name)
                events.append({"log": str(path), "output_dir": current_output, **rec})
                continue
            if event:
                if "nonfinite" in event or "grad_health" in event or event.startswith("runtime"):
                    events.append({"log": str(path), "output_dir": current_output, **rec})
                continue
            if "step" not in rec:
                continue
            metric_keys = [k for k in rec if "/" in k]
            if not metric_keys:
                continue
            row = {
                "log": str(path.relative_to(ROOT)),
                "output_dir": current_output or "",
                "run_id": current_run_id or "",
                "run_name": current_run_name or "",
                "step": rec["step"],
            }
            for key in metric_keys:
                row[key] = rec[key]
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df.dropna(subset=["step"])
        df = df.sort_values(["output_dir", "step", "log"])
        # Keep the latest duplicate metric row per output/step/logical metric merge.
        df = df.groupby(["output_dir", "step"], as_index=False).last()
    return df, pd.DataFrame(events)


def records_for(df: pd.DataFrame, run_dirs: list[str], metric: str) -> dict[str, pd.DataFrame]:
    out = {}
    for run_dir in run_dirs:
        sub = df[(df["output_dir"] == run_dir) & df[metric].notna()][["step", metric]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("step")
        sub = sub.groupby("step", as_index=False).last()
        out[RUN_LABELS.get(run_dir, Path(run_dir).name)] = sub
    return out


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def plot_lines(
    series: dict[str, pd.DataFrame],
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    smooth: bool = False,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=180)
    for label, sub in series.items():
        x = sub["step"].to_numpy(dtype=float) / 1000.0
        y = sub[metric].to_numpy(dtype=float)
        if smooth and len(y) > 4:
            y = ema(y)
        ax.plot(x, y, linewidth=1.8, label=label)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(fontsize=7.2, frameon=False)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_instability(df: pd.DataFrame, events: pd.DataFrame) -> None:
    run_dir = "outputs/simamba_2day_50m_seq128_100m_lr1e4_20260502_025645"
    sub = df[df["output_dir"] == run_dir].sort_values("step")
    sub = sub[sub["step"] <= 170]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    ax2 = ax1.twinx()
    ax1.plot(sub["step"], sub["train/loss"], color="#2563eb", label="train/loss", linewidth=1.6)
    ax2.plot(
        sub["step"],
        sub["train/grad_norm_pre_clip"],
        color="#dc2626",
        label="grad norm pre-clip",
        linewidth=1.4,
        alpha=0.9,
    )
    nonfinite = events[
        (events.get("output_dir", "") == run_dir)
        & (events.get("event", "").isin(["train.nonfinite_detected", "train.grad_health"]))
    ]
    if not nonfinite.empty:
        step = float(nonfinite["step"].dropna().iloc[-1])
        ax1.axvline(step, color="#111827", linestyle="--", linewidth=1.0)
        ax1.text(step + 1, ax1.get_ylim()[0] + 0.05, "nonfinite", fontsize=7, rotation=90)
    ax1.set_title("Initial unstable run: clipping did not fix nonfinite gradients", fontsize=11, weight="bold")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training loss", color="#2563eb")
    ax2.set_ylabel("Pre-clip gradient norm", color="#dc2626")
    ax1.grid(True, color="#e5e7eb", linewidth=0.8)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
    ax1.legend(lines[: len(labels)], labels, fontsize=7.4, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "wandb_initial_instability.png", bbox_inches="tight")
    plt.close(fig)


def best_val_from_outputs() -> pd.DataFrame:
    rows = []
    for metric_file in (ROOT / "outputs").rglob("best/metrics.json"):
        rel = str(metric_file.parent.parent.relative_to(ROOT))
        if rel not in RUN_LABELS:
            continue
        try:
            metrics = json.loads(metric_file.read_text())
        except json.JSONDecodeError:
            continue
        val = metrics.get("val/loss")
        if val is None:
            continue
        rows.append({"run": RUN_LABELS[rel], "output_dir": rel, "best_val": float(val), "step": metrics.get("step")})
    return pd.DataFrame(rows)


def plot_best_val() -> None:
    df = best_val_from_outputs()
    order = [RUN_LABELS[r] for r in MAIN_500M + ABLATION_50M + LOCAL_MIX if r in RUN_LABELS]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    df = df.sort_values("run")
    colors = ["#2563eb" if "Mamba2" in r else "#0891b2" if "Simamba" in r else "#f97316" for r in df["run"].astype(str)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    ax.barh(df["run"].astype(str), df["best_val"], color=colors)
    ax.set_xlabel("Best fixed validation loss")
    ax.set_title("Best validation loss across controlled runs", fontsize=11, weight="bold")
    ax.grid(True, axis="x", color="#e5e7eb")
    for i, val in enumerate(df["best_val"]):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=7.2)
    ax.set_xlim(4.75, max(df["best_val"]) + 0.25)
    fig.tight_layout()
    fig.savefig(OUT / "summary_best_val_loss.png", bbox_inches="tight")
    plt.close(fig)


def plot_throughput(df: pd.DataFrame) -> None:
    rows = []
    for run_dir in MAIN_500M + LOCAL_MIX:
        sub = df[(df["output_dir"] == run_dir) & df["train/tokens_per_sec"].notna()].sort_values("step")
        if sub.empty:
            continue
        vals = sub["train/tokens_per_sec"].tail(100).to_numpy(dtype=float)
        rows.append({"run": RUN_LABELS[run_dir], "tokens_per_sec": float(np.median(vals))})
    tdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=180)
    ax.barh(tdf["run"], tdf["tokens_per_sec"], color="#0f766e")
    ax.set_xlabel("Median tail throughput (tokens/s)")
    ax.set_title("Training throughput on Tesla V100", fontsize=11, weight="bold")
    ax.grid(True, axis="x", color="#e5e7eb")
    for i, val in enumerate(tdf["tokens_per_sec"]):
        ax.text(val + 200, i, f"{val:,.0f}", va="center", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(OUT / "wandb_training_throughput.png", bbox_inches="tight")
    plt.close(fig)


def plot_compression() -> None:
    path = ROOT / "run_logs" / "compression_eval_20260504.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    base = df[df["variant"] == "baseline"][["checkpoint", "loss"]].rename(columns={"loss": "base_loss"})
    df = df.merge(base, on="checkpoint")
    df["delta"] = df["loss"] - df["base_loss"]
    keep = ["int8_row_qdq", "int4_row_qdq", "prune_global_10", "prune_global_20", "prune_global_30"]
    df = df[df["variant"].isin(keep)]
    labels = {
        "int8_row_qdq": "int8 qdq",
        "int4_row_qdq": "int4 qdq",
        "prune_global_10": "10% prune",
        "prune_global_20": "20% prune",
        "prune_global_30": "30% prune",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=180)
    x = np.arange(len(keep))
    width = 0.24
    checkpoints = ["mamba2_500m", "simamba_midpoint_500m", "simamba_trapezoid_500m"]
    for i, ckpt in enumerate(checkpoints):
        sub = df[df["checkpoint"] == ckpt].set_index("variant").loc[keep]
        ax.bar(x + (i - 1) * width, sub["delta"], width=width, label=ckpt.replace("_", " "))
    ax.set_xticks(x, [labels[k] for k in keep], rotation=15, ha="right")
    ax.set_ylabel("Validation loss delta")
    ax.set_title("Post-training perturbation sensitivity", fontsize=11, weight="bold")
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.legend(fontsize=7.2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "compression_loss_delta.png", bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec="#1f2937", fontsize=8.5):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, start, end, text=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#374151"))
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=7, color="#374151")


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.7), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "Simamba language-model block", ha="center", fontsize=13, weight="bold")
    ax.text(
        0.5,
        0.91,
        "Mamba-3 SISO structure with Simpson or matched trapezoid state update",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    add_box(ax, (0.04, 0.76), (0.13, 0.08), "Token IDs", "#eef2ff", "#4338ca")
    add_box(ax, (0.23, 0.76), (0.14, 0.08), "Embedding", "#eef2ff", "#4338ca")
    add_box(ax, (0.43, 0.76), (0.17, 0.08), "Add + RMSNorm", "#f8fafc")
    add_box(ax, (0.68, 0.76), (0.17, 0.08), "Simamba mixer", "#ecfeff", "#0891b2")
    add_box(ax, (0.41, 0.61), (0.18, 0.08), "Linear projection", "#f8fafc")
    add_box(ax, (0.09, 0.46), (0.16, 0.08), "$z$ gate", "#fff7ed", "#c2410c")
    add_box(ax, (0.35, 0.46), (0.18, 0.08), "$x,B,C$ stream", "#ecfeff", "#0891b2")
    add_box(ax, (0.66, 0.46), (0.18, 0.08), "$\\Delta,A,c,\\theta$", "#fefce8", "#a16207")
    add_box(ax, (0.32, 0.31), (0.24, 0.08), "Optional depthwise causal\nConv1d over $x/B/C$", "#dcfce7", "#15803d", 7.8)
    add_box(ax, (0.25, 0.16), (0.20, 0.10), "RMSNormGated\non $B,C$", "#f8fafc")
    add_box(ax, (0.53, 0.14), (0.22, 0.13), "SISO recurrence\nSimpson width-3 or\ntrapezoid width-2", "#e0f2fe", "#0369a1", 8.0)
    add_box(ax, (0.80, 0.15), (0.17, 0.10), "$D$ skip, $z$ gate,\nout projection", "#fff7ed", "#c2410c", 8.0)
    add_arrow(ax, (0.17, 0.80), (0.23, 0.80))
    add_arrow(ax, (0.37, 0.80), (0.43, 0.80))
    add_arrow(ax, (0.60, 0.80), (0.68, 0.80))
    add_arrow(ax, (0.765, 0.76), (0.50, 0.69))
    add_arrow(ax, (0.47, 0.61), (0.17, 0.54))
    add_arrow(ax, (0.50, 0.61), (0.44, 0.54))
    add_arrow(ax, (0.54, 0.61), (0.75, 0.54))
    add_arrow(ax, (0.44, 0.46), (0.44, 0.39))
    add_arrow(ax, (0.44, 0.31), (0.44, 0.26))
    add_arrow(ax, (0.45, 0.21), (0.53, 0.21))
    add_arrow(ax, (0.75, 0.46), (0.66, 0.27))
    add_arrow(ax, (0.75, 0.205), (0.80, 0.205))
    ax.text(
        0.04,
        0.03,
        "The matched trapezoid baseline reuses the same projections, biases, rotary state, D skip, and output path.",
        fontsize=7.2,
        color="#374151",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "simamba_architecture.png", bbox_inches="tight")
    plt.close(fig)


def plot_discretization_comparison() -> None:
    paths = [
        (ROOT / "docs" / "assets" / "eulers_discretization.png", "(a) Forward Euler", "left endpoint only", 145),
        (ROOT / "docs" / "assets" / "trap_discretization.png", "(b) Trapezoid", "two endpoint average", 92),
        (ROOT / "docs" / "assets" / "simpson_discretization.png", "(c) Simpson's 1/3 rule", "endpoint + midpoint curvature", 92),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), dpi=220)
    for ax, (path, title, subtitle, crop_top) in zip(axes, paths):
        image = plt.imread(path)
        # Remove the original title band so the paper has consistent labels and
        # avoids carrying the typo in the provided Simpson source image.
        cropped = image[crop_top:735, :, :]
        ax.imshow(cropped)
        ax.axis("off")
        ax.set_title(title, fontsize=11, weight="bold", pad=4)
        ax.text(0.5, -0.045, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    fig.suptitle("Quadrature choices for one SSM input-forcing interval", y=0.99, fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.35)
    fig.savefig(OUT / "discretization_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _trim_near_white(image: Image.Image, threshold: int = 248) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    pad = 24
    left = max(0, xs.min() - pad)
    right = min(rgb.width, xs.max() + pad)
    top = max(0, ys.min() - pad)
    bottom = min(rgb.height, ys.max() + pad)
    return rgb.crop((left, top, right, bottom))


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas_w, canvas_h = size
    scale = min(canvas_w / image.width, canvas_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (canvas_w - resized.width) // 2
    y = (canvas_h - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def plot_mamba_family_architecture_comparison() -> None:
    mamba23 = Image.open(ROOT / "assets" / "mamba3.png").convert("RGB")
    # The README image includes a large legend on the right. Crop to the
    # Mamba-2/Mamba-3 block diagrams so the block comparison remains legible.
    mamba23 = _trim_near_white(mamba23.crop((0, 0, 2100, mamba23.height)))
    simamba = _trim_near_white(Image.open(ROOT / "docs" / "assets" / "simamba_architecture.png"))

    panel_w, panel_h = 1500, 900
    margin = 80
    title_h = 110
    caption_h = 70
    gap = 50
    canvas = Image.new("RGB", (2 * panel_w + gap + 2 * margin, panel_h + title_h + caption_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        note_font = ImageFont.truetype("DejaVuSans.ttf", 27)
    except OSError:
        title_font = label_font = note_font = ImageFont.load_default()

    title = "Mamba-family block diagrams"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) // 2, 26), title, fill="black", font=title_font)

    left = _fit_image(mamba23, (panel_w, panel_h))
    right = _fit_image(simamba, (panel_w, panel_h))
    x0 = margin
    x1 = margin + panel_w + gap
    y0 = margin + title_h
    canvas.paste(left, (x0, y0))
    canvas.paste(right, (x1, y0))

    labels = [
        ("Mamba-2 and Mamba-3 reference blocks", x0, "README: assets/mamba3.png"),
        ("Simamba block", x1, "docs/assets/simamba_architecture.png"),
    ]
    for label, x, note in labels:
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (panel_w - (bbox[2] - bbox[0])) // 2, y0 + panel_h + 10), label, fill="black", font=label_font)
        bbox = draw.textbbox((0, 0), note, font=note_font)
        draw.text((x + (panel_w - (bbox[2] - bbox[0])) // 2, y0 + panel_h + 52), note, fill="#374151", font=note_font)

    canvas.save(OUT / "mamba_family_architecture_comparison.png")


def write_run_summary(df: pd.DataFrame) -> None:
    rows = []
    best = best_val_from_outputs().set_index("output_dir")
    for run_dir, label in RUN_LABELS.items():
        sub = df[df["output_dir"] == run_dir].sort_values("step")
        if sub.empty:
            continue
        last_train = sub[sub["train/loss"].notna()]["train/loss"].tail(1)
        last_val = sub[sub["val/loss"].notna()]["val/loss"].tail(1)
        grad = sub[sub["train/grad_norm_pre_clip"].notna()]["train/grad_norm_pre_clip"]
        throughput = sub[sub["train/tokens_per_sec"].notna()]["train/tokens_per_sec"]
        rows.append(
            {
                "run": label,
                "output_dir": run_dir,
                "best_val": best.loc[run_dir]["best_val"] if run_dir in best.index else math.nan,
                "best_step": best.loc[run_dir]["step"] if run_dir in best.index else math.nan,
                "last_val": float(last_val.iloc[-1]) if not last_val.empty else math.nan,
                "last_train": float(last_train.iloc[-1]) if not last_train.empty else math.nan,
                "median_tail_tokens_per_sec": float(np.median(throughput.tail(100))) if not throughput.empty else math.nan,
                "median_grad_norm_pre_clip": float(np.median(grad.tail(100))) if not grad.empty else math.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "run_summary.csv", index=False)


def copy_benchmark_plots() -> None:
    src = ROOT / "benchmarks" / "plots"
    for name in ["simamba_latency.png", "simamba_throughput.png", "simamba_memory.png", "simamba_ratio_summary.png"]:
        if (src / name).exists():
            shutil.copy2(src / name, OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    df, events = parse_run_logs()
    df.to_csv(OUT / "wandb_local_history.csv", index=False)
    events.to_csv(OUT / "run_events.csv", index=False)
    write_run_summary(df)

    plot_lines(records_for(df, MAIN_500M, "val/loss"), "val/loss", "wandb_main_500m_val_loss.png", "500M-token comparison: fixed validation loss", "Validation loss", ylim=(4.75, 6.2))
    plot_lines(records_for(df, MAIN_500M, "train/loss"), "train/loss", "wandb_main_500m_train_loss.png", "500M-token comparison: training loss", "Training loss", smooth=True, ylim=(4.0, 11.2))
    plot_lines(records_for(df, MAIN_500M, "train/grad_norm_pre_clip"), "train/grad_norm_pre_clip", "wandb_main_500m_grad_norm.png", "500M-token comparison: pre-clip gradient norm", "Gradient norm", smooth=True, ylim=(0, 8.0))
    plot_lines(records_for(df, ABLATION_50M, "val/loss"), "val/loss", "wandb_50m_discretization_val_loss.png", "50M-token discretization ablation", "Validation loss", ylim=(5.75, 8.2))
    plot_lines(records_for(df, LOCAL_MIX, "val/loss"), "val/loss", "wandb_localmix_val_loss.png", "Local mixing controls on 50M tokens", "Validation loss", ylim=(5.75, 8.2))
    plot_instability(df, events)
    plot_best_val()
    plot_throughput(df)
    plot_compression()
    plot_architecture()
    plot_discretization_comparison()
    plot_mamba_family_architecture_comparison()
    copy_benchmark_plots()
    print(OUT)


if __name__ == "__main__":
    main()
