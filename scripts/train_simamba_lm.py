#!/usr/bin/env python

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Mamba-family LM pretraining script.")
    parser.add_argument("--train-data", type=Path, required=True, help="Tokenized train .bin file.")
    parser.add_argument("--val-data", type=Path, default=None, help="Optional tokenized val .bin file.")
    parser.add_argument("--token-dtype", choices=["uint16", "int32", "int64"], default="uint16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--model-layer", choices=["Mamba1", "Mamba2", "Mamba3", "Simamba"], default="Simamba")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--d-intermediate", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=50280)
    parser.add_argument("--pad-vocab-size-multiple", type=int, default=8)
    parser.add_argument("--rms-norm", action="store_true", default=True)
    parser.add_argument("--no-rms-norm", dest="rms_norm", action="store_false")
    parser.add_argument("--fused-add-norm", action="store_true", default=True)
    parser.add_argument("--no-fused-add-norm", dest="fused_add_norm", action="store_false")
    parser.add_argument("--residual-in-fp32", action="store_true", default=True)
    parser.add_argument("--no-residual-in-fp32", dest="residual_in_fp32", action="store_false")

    parser.add_argument("--simamba-d-state", type=int, default=128)
    parser.add_argument("--simamba-expand", type=int, default=2)
    parser.add_argument("--simamba-headdim", type=int, default=64)
    parser.add_argument("--simamba-ngroups", type=int, default=1)
    parser.add_argument("--simamba-rope-fraction", type=float, default=0.5)
    parser.add_argument("--simamba-chunk-size", type=int, default=64)
    parser.add_argument("--simamba-use-midpoint-control", action="store_true")
    parser.add_argument("--simamba-backend", choices=["reference", "triton"], default="triton")
    parser.add_argument("--simamba-outproj-norm", action="store_true")
    parser.add_argument("--mamba2-d-state", type=int, default=128)
    parser.add_argument("--mamba2-d-conv", type=int, default=4)
    parser.add_argument("--mamba2-expand", type=int, default=2)
    parser.add_argument("--mamba2-headdim", type=int, default=64)
    parser.add_argument("--mamba2-ngroups", type=int, default=1)
    parser.add_argument("--mamba2-chunk-size", type=int, default=256)
    parser.add_argument("--mamba2-use-mem-eff-path", action="store_true", default=True)
    parser.add_argument("--no-mamba2-use-mem-eff-path", dest="mamba2_use_mem_eff_path", action="store_false")
    parser.add_argument("--mamba3-d-state", type=int, default=128)
    parser.add_argument("--mamba3-expand", type=int, default=2)
    parser.add_argument("--mamba3-headdim", type=int, default=64)
    parser.add_argument("--mamba3-ngroups", type=int, default=1)
    parser.add_argument("--mamba3-rope-fraction", type=float, default=0.5)
    parser.add_argument("--mamba3-chunk-size", type=int, default=64)
    parser.add_argument("--mamba3-outproj-norm", action="store_true")
    parser.add_argument("--mamba3-is-mimo", action="store_true")
    parser.add_argument("--mamba3-mimo-rank", type=int, default=4)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-milestones", type=int, default=0, help="Max number of milestone checkpoints to keep.")
    parser.add_argument("--save-optimizer-latest-only", action="store_true", default=True)
    parser.add_argument("--save-optimizer-all", dest="save_optimizer_latest_only", action="store_false")
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="simamba-pretrain")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)

    parser.add_argument("--lm-eval-every", type=int, default=0, help="Run lm-eval every N steps. 0 disables.")
    parser.add_argument("--lm-eval-tasks", default="lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa")
    parser.add_argument("--lm-eval-batch-size", type=int, default=64)
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_master(rank):
    return rank == 0


class PackedTokenDataset:
    def __init__(self, path: Path, dtype_name: str):
        dtype_map = {"uint16": np.uint16, "int32": np.int32, "int64": np.int64}
        self.tokens = np.memmap(path, mode="r", dtype=dtype_map[dtype_name])
        self.size = int(self.tokens.shape[0])

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device):
        max_start = self.size - seq_len - 1
        if max_start <= 0:
            raise ValueError(f"Dataset {self.size} too small for seq_len={seq_len}.")
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.empty((batch_size, seq_len), dtype=torch.long)
        y = torch.empty((batch_size, seq_len), dtype=torch.long)
        for i, start in enumerate(starts.tolist()):
            chunk = np.asarray(self.tokens[start : start + seq_len + 1], dtype=np.int64)
            x[i] = torch.from_numpy(chunk[:-1].copy())
            y[i] = torch.from_numpy(chunk[1:].copy())
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def make_config(args):
    if args.model_layer == "Simamba":
        ssm_cfg = {
            "layer": "Simamba",
            "d_state": args.simamba_d_state,
            "expand": args.simamba_expand,
            "headdim": args.simamba_headdim,
            "ngroups": args.simamba_ngroups,
            "rope_fraction": args.simamba_rope_fraction,
            "chunk_size": args.simamba_chunk_size,
            "use_midpoint_control": args.simamba_use_midpoint_control,
            "simamba_backend": args.simamba_backend,
            "is_outproj_norm": args.simamba_outproj_norm,
        }
    elif args.model_layer == "Mamba2":
        ssm_cfg = {
            "layer": "Mamba2",
            "d_state": args.mamba2_d_state,
            "d_conv": args.mamba2_d_conv,
            "expand": args.mamba2_expand,
            "headdim": args.mamba2_headdim,
            "ngroups": args.mamba2_ngroups,
            "chunk_size": args.mamba2_chunk_size,
            "use_mem_eff_path": args.mamba2_use_mem_eff_path,
        }
    elif args.model_layer == "Mamba3":
        ssm_cfg = {
            "layer": "Mamba3",
            "d_state": args.mamba3_d_state,
            "expand": args.mamba3_expand,
            "headdim": args.mamba3_headdim,
            "ngroups": args.mamba3_ngroups,
            "rope_fraction": args.mamba3_rope_fraction,
            "chunk_size": args.mamba3_chunk_size,
            "is_outproj_norm": args.mamba3_outproj_norm,
            "is_mimo": args.mamba3_is_mimo,
            "mimo_rank": args.mamba3_mimo_rank,
        }
    else:
        ssm_cfg = {"layer": "Mamba1"}
    return MambaConfig(
        d_model=args.d_model,
        d_intermediate=args.d_intermediate,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        ssm_cfg=ssm_cfg,
        rms_norm=args.rms_norm,
        residual_in_fp32=args.residual_in_fp32,
        fused_add_norm=args.fused_add_norm,
        pad_vocab_size_multiple=args.pad_vocab_size_multiple,
    )


def get_dtype(dtype_name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]


def lr_for_step(step: int, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def save_checkpoint(path: Path, model, optimizer, step: int, args, include_optimizer: bool):
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": raw_model.state_dict(),
        "step": step,
        "args": vars(args),
        "config": asdict(raw_model.config),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


@torch.no_grad()
def evaluate(model, dataset, args, device, amp_ctx):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = dataset.sample_batch(args.micro_batch_size, args.seq_len, device)
        with amp_ctx():
            logits = model(x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.detach())
    model.train()
    return torch.stack(losses).mean()


def maybe_run_lm_eval(args, checkpoint_dir: Path, step: int):
    if args.lm_eval_every <= 0 or step % args.lm_eval_every != 0:
        return None
    cmd = [
        "python",
        "evals/lm_harness_eval.py",
        "--model",
        "mamba",
        "--model_args",
        f"pretrained={checkpoint_dir}",
        "--tasks",
        args.lm_eval_tasks,
        "--device",
        args.device,
        "--batch_size",
        str(args.lm_eval_batch_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (checkpoint_dir / "lm_eval_stdout.txt").write_text(proc.stdout)
    (checkpoint_dir / "lm_eval_stderr.txt").write_text(proc.stderr)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def replace_dir(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def list_milestones(output_dir: Path):
    if not output_dir.exists():
        return []
    items = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("step_"):
            items.append(child)
    return sorted(items, key=lambda p: p.name)


def prune_milestones(output_dir: Path, keep: int):
    if keep <= 0:
        for path in list_milestones(output_dir):
            shutil.rmtree(path)
        return
    milestones = list_milestones(output_dir)
    for path in milestones[:-keep]:
        shutil.rmtree(path)


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    train_data = PackedTokenDataset(args.train_data, args.token_dtype)
    val_data = PackedTokenDataset(args.val_data, args.token_dtype) if args.val_data is not None else None

    config = make_config(args)
    model = MambaLMHeadModel(config=config, device=device, dtype=get_dtype(args.dtype))
    if args.compile:
        model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16" and device.type == "cuda"))
    amp_dtype = get_dtype(args.dtype)
    amp_ctx = (
        lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
        if args.dtype != "fp32"
        else nullcontext
    )

    wandb_run = None
    if args.wandb and is_master(rank):
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            config=vars(args),
        )

    tokens_per_step = args.global_batch_size * args.seq_len
    last_log_time = time.time()
    best_val_loss: Optional[float] = None

    for step in range(start_step, args.max_steps):
        lr = lr_for_step(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for _ in range(grad_accum_steps):
            x, y = train_data.sample_batch(args.micro_batch_size, args.seq_len, device)
            with amp_ctx():
                logits = model(x).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.zeros((), device=device)

        scaler.step(optimizer)
        scaler.update()

        if distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        if is_master(rank) and step % args.log_every == 0:
            now = time.time()
            dt = max(now - last_log_time, 1e-6)
            last_log_time = now
            tokens_per_sec = tokens_per_step * args.log_every / dt if step > 0 else 0.0
            metrics = {
                "step": step,
                "train/loss": float(loss_accum.item()),
                "train/lr": lr,
                "train/grad_norm": float(grad_norm.item()),
                "train/tokens_per_sec": tokens_per_sec,
            }
            print(json.dumps(metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        if val_data is not None and step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args, device, amp_ctx)
            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            if is_master(rank):
                metrics = {"step": step, "val/loss": float(val_loss.item())}
                print(json.dumps(metrics), flush=True)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                if args.save_best and (best_val_loss is None or float(val_loss.item()) < best_val_loss):
                    best_val_loss = float(val_loss.item())
                    best_dir = args.output_dir / "best"
                    tmp_dir = args.output_dir / ".best_build"
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        tmp_dir / "trainer.pt",
                        model,
                        optimizer,
                        step,
                        args,
                        include_optimizer=False,
                    )
                    raw_model = model.module if isinstance(model, DDP) else model
                    raw_model.save_pretrained(tmp_dir)
                    (tmp_dir / "metrics.json").write_text(json.dumps({"step": step, "val/loss": best_val_loss}, indent=2))
                    replace_dir(tmp_dir, best_dir)
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        if is_master(rank) and step > 0 and step % args.save_every == 0:
            latest_dir = args.output_dir / "latest"
            tmp_latest = args.output_dir / ".latest_build"
            if tmp_latest.exists():
                shutil.rmtree(tmp_latest)
            tmp_latest.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                tmp_latest / "trainer.pt",
                model,
                optimizer,
                step,
                args,
                include_optimizer=args.save_optimizer_latest_only,
            )
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.save_pretrained(tmp_latest)
            replace_dir(tmp_latest, latest_dir)
            shutil.rmtree(tmp_latest, ignore_errors=True)

            milestone_dir = args.output_dir / f"step_{step:07d}"
            if args.keep_milestones > 0:
                if milestone_dir.exists():
                    shutil.rmtree(milestone_dir)
                shutil.copytree(latest_dir, milestone_dir)
                trainer_path = milestone_dir / "trainer.pt"
                if trainer_path.exists() and args.save_optimizer_latest_only:
                    ckpt = torch.load(trainer_path, map_location="cpu")
                    ckpt.pop("optimizer", None)
                    torch.save(ckpt, trainer_path)
                prune_milestones(args.output_dir, args.keep_milestones)

            lm_eval_result = maybe_run_lm_eval(args, latest_dir, step)
            if lm_eval_result is not None:
                summary = {
                    "step": step,
                    "lm_eval/returncode": lm_eval_result["returncode"],
                }
                print(json.dumps(summary), flush=True)
                if wandb_run is not None:
                    wandb_run.log(summary, step=step)

    if is_master(rank):
        ckpt_dir = args.output_dir / "latest"
        tmp_dir = args.output_dir / ".final_build"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            tmp_dir / "trainer.pt",
            model,
            optimizer,
            args.max_steps,
            args,
            include_optimizer=args.save_optimizer_latest_only,
        )
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.save_pretrained(tmp_dir)
        replace_dir(tmp_dir, ckpt_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
