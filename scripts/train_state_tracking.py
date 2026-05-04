#!/usr/bin/env python3
"""Train small Mamba-family classifiers on synthetic state-tracking tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel


def emit(event: str | None = None, **payload):
    row = dict(payload)
    if event is not None:
        row["event"] = event
    row["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
    print(json.dumps(row), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["parity", "mod_sum", "signed_mod"], default="parity")
    parser.add_argument("--modulus", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-seq-lens", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--model-layer", choices=["Mamba2", "Simamba"], default="Simamba")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--simamba-backend", choices=["reference", "triton"], default="triton")
    parser.add_argument("--simamba-discretization", choices=["simpson", "trapezoid"], default="simpson")
    parser.add_argument("--simamba-a-max", type=float, default=4.0)
    parser.add_argument("--simamba-outproj-norm", action="store_true")
    parser.add_argument("--mamba2-use-mem-eff-path", action="store_true", default=True)
    parser.add_argument("--no-mamba2-use-mem-eff-path", dest="mamba2_use_mem_eff_path", action="store_false")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="simamba")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def task_sizes(task: str, modulus: int):
    if task == "parity":
        return 2, 2
    if task == "mod_sum":
        return modulus, modulus
    if task == "signed_mod":
        return 2, modulus
    raise ValueError(task)


def sample_batch(task: str, batch_size: int, seq_len: int, modulus: int, device: torch.device):
    if task == "parity":
        x = torch.randint(0, 2, (batch_size, seq_len), device=device)
        y = x.sum(dim=1).remainder(2)
    elif task == "mod_sum":
        x = torch.randint(0, modulus, (batch_size, seq_len), device=device)
        y = x.sum(dim=1).remainder(modulus)
    elif task == "signed_mod":
        x = torch.randint(0, 2, (batch_size, seq_len), device=device)
        signed = x.mul(2).sub(1)
        y = signed.sum(dim=1).remainder(modulus)
    else:
        raise ValueError(task)
    return x.long(), y.long()


def make_ssm_cfg(args):
    if args.model_layer == "Mamba2":
        return {
            "layer": "Mamba2",
            "d_state": args.d_state,
            "d_conv": 4,
            "expand": args.expand,
            "headdim": args.headdim,
            "ngroups": 1,
            "chunk_size": args.chunk_size,
            "use_mem_eff_path": args.mamba2_use_mem_eff_path,
        }
    if args.simamba_discretization == "trapezoid" and args.simamba_backend != "reference":
        raise ValueError("Trapezoid Simamba baseline currently requires --simamba-backend reference.")
    return {
        "layer": "Simamba",
        "d_state": args.d_state,
        "expand": args.expand,
        "headdim": args.headdim,
        "rope_fraction": 0.5,
        "chunk_size": args.chunk_size,
        "recompute_chunk_size": args.chunk_size,
        "dt_limit": (0.001, 0.1),
        "A_max": args.simamba_a_max,
        "use_midpoint_control": False,
        "discretization": args.simamba_discretization,
        "simamba_backend": args.simamba_backend,
        "is_outproj_norm": args.simamba_outproj_norm,
    }


class StateTrackingModel(nn.Module):
    def __init__(self, args, vocab_size: int, num_classes: int, device: torch.device):
        super().__init__()
        config = MambaConfig(
            d_model=args.d_model,
            n_layer=args.n_layer,
            vocab_size=vocab_size,
            ssm_cfg=make_ssm_cfg(args),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            pad_vocab_size_multiple=8,
        )
        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            device=device,
            dtype=torch.float32,
        )
        self.classifier = nn.Linear(args.d_model, num_classes, device=device, dtype=torch.float32)

    def forward(self, input_ids):
        hidden = self.backbone(input_ids)
        return self.classifier(hidden[:, -1])


@torch.no_grad()
def evaluate(model, args, seq_len: int, device: torch.device, amp_ctx):
    model.eval()
    losses = []
    correct = 0
    total = 0
    for _ in range(args.eval_batches):
        x, y = sample_batch(args.task, args.batch_size, seq_len, args.modulus, device)
        with amp_ctx():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return sum(losses) / len(losses), correct / total


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    vocab_size, num_classes = task_sizes(args.task, args.modulus)
    eval_seq_lens = args.eval_seq_lens or [args.seq_len, args.seq_len * 2]
    model = StateTrackingModel(args, vocab_size, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    amp_ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if amp_dtype is not None else nullcontext

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )

    emit(
        "startup",
        task=args.task,
        model_layer=args.model_layer,
        simamba_discretization=args.simamba_discretization if args.model_layer == "Simamba" else None,
        simamba_backend=args.simamba_backend if args.model_layer == "Simamba" else None,
        param_count=sum(p.numel() for p in model.parameters()),
        vocab_size=vocab_size,
        num_classes=num_classes,
    )

    last_metrics = {}
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            metrics = {}
            for eval_len in eval_seq_lens:
                loss, acc = evaluate(model, args, eval_len, device, amp_ctx)
                metrics[f"eval/loss_len{eval_len}"] = loss
                metrics[f"eval/acc_len{eval_len}"] = acc
            metrics["step"] = step
            emit(None, **metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            last_metrics = metrics
            if step == args.steps:
                break

        x, y = sample_batch(args.task, args.batch_size, args.seq_len, args.modulus, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            pred = logits.argmax(dim=-1)
            acc = float((pred == y).float().mean().item())
            row = {
                "step": step,
                "train/loss": float(loss.item()),
                "train/acc": acc,
                "train/grad_norm_pre_clip": float(grad_norm.item()),
                "train/lr": args.lr,
            }
            emit(None, **row)
            if wandb_run is not None:
                wandb_run.log(row, step=step)

    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(last_metrics, f, indent=2, sort_keys=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
