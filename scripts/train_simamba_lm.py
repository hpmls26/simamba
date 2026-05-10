#!/usr/bin/env python

import argparse
import json
import math
import os
import signal
import shutil
import subprocess
import sys
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


def emit_status(event: str, *, rank: Optional[int] = None, **fields):
    if rank not in (None, 0):
        return
    payload = {
        "event": event,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
    }
    if rank is not None:
        payload["rank"] = rank
    payload.update(fields)
    print(json.dumps(payload), flush=True)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_optimizer(model, args):
    decay_params = []
    no_decay_params = []
    decay_tensors = 0
    no_decay_tensors = 0
    decay_elements = 0
    no_decay_elements = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_no_decay = (
            getattr(parameter, "_no_weight_decay", False)
            or parameter.ndim <= 1
            or name.endswith("bias")
        )
        if use_no_decay:
            no_decay_params.append(parameter)
            no_decay_tensors += 1
            no_decay_elements += parameter.numel()
        else:
            decay_params.append(parameter)
            decay_tensors += 1
            decay_elements += parameter.numel()

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=tuple(args.betas),
    )
    stats = {
        "decay_tensors": decay_tensors,
        "no_decay_tensors": no_decay_tensors,
        "decay_elements": decay_elements,
        "no_decay_elements": no_decay_elements,
    }
    return optimizer, stats


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
    parser.add_argument("--simamba-recompute-chunk-size", type=int, default=None)
    parser.add_argument("--simamba-dt-limit", type=float, nargs=2, default=(1e-3, 0.1))
    parser.add_argument(
        "--simamba-a-max",
        type=float,
        default=None,
        help=(
            "Upper bound for -A in Simamba dynamics. Defaults to 4.0 for "
            "Simamba Triton fp16 CUDA runs to avoid V100 fp16 scan overflow, "
            "and 16.0 otherwise."
        ),
    )
    parser.add_argument(
        "--simamba-d-conv",
        type=int,
        default=0,
        help=(
            "Depthwise causal local convolution width over the concatenated Simamba x/B/C stream. "
            "0 preserves the original no-local-conv Simamba block; 4 matches the default Mamba2 local mixing width."
        ),
    )
    parser.add_argument("--simamba-use-midpoint-control", action="store_true")
    parser.add_argument(
        "--simamba-control-logit-offset",
        type=float,
        default=0.0,
        help=(
            "Fixed logit offset added before sigmoid for the Simamba Simpson/trapezoid "
            "control projection. Negative values initialize the control closer to zero."
        ),
    )
    parser.add_argument(
        "--simamba-midpoint-logit-offset",
        type=float,
        default=0.0,
        help=(
            "Fixed logit offset added before sigmoid for the optional Simamba midpoint "
            "control projection."
        ),
    )
    parser.add_argument(
        "--simamba-correction-anneal-min",
        type=float,
        default=1.0,
        help=(
            "Initial lower bound for the effective Simpson correction scale. "
            "Use 0.0 to remove the negative lag-2 correction at the start of training."
        ),
    )
    parser.add_argument(
        "--simamba-correction-anneal-max",
        type=float,
        default=1.0,
        help="Final effective Simpson correction scale after annealing.",
    )
    parser.add_argument(
        "--simamba-correction-anneal-start",
        type=int,
        default=0,
        help="Optimizer step at which to begin ramping the Simpson correction scale.",
    )
    parser.add_argument(
        "--simamba-correction-anneal-steps",
        type=int,
        default=0,
        help="Number of optimizer steps used to ramp the Simpson correction scale from min to max.",
    )
    parser.add_argument(
        "--simamba-correction-anneal-schedule",
        choices=["linear", "cosine"],
        default="linear",
        help="Schedule shape for --simamba-correction-anneal-steps.",
    )
    parser.add_argument("--simamba-discretization", choices=["simpson", "trapezoid"], default="simpson")
    parser.add_argument("--simamba-backend", choices=["reference", "triton", "improved"], default="triton")
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
    parser.add_argument(
        "--train-sampling",
        choices=["epoch", "random"],
        default="epoch",
        help=(
            "Training sample order. 'epoch' uses shuffled non-overlapping token blocks "
            "without replacement within each pass through the .bin file; 'random' keeps "
            "the previous with-replacement random-span sampling."
        ),
    )
    parser.add_argument(
        "--eval-sampling",
        choices=["fixed", "random"],
        default="fixed",
        help=(
            "Validation sample order. 'fixed' evaluates the same deterministic token spans "
            "every time; 'random' keeps the previous random-span validation."
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed for fixed validation spans. Defaults to --seed + 100000.",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--grad-scale-init",
        type=float,
        default=8192.0,
        help="Initial CUDA GradScaler scale for fp16 training.",
    )
    parser.add_argument(
        "--grad-scale-growth-interval",
        type=int,
        default=2000,
        help="GradScaler growth interval for fp16 training.",
    )
    parser.add_argument(
        "--skip-nonfinite-steps",
        action="store_true",
        help="Skip optimizer updates with nonfinite loss/gradients instead of aborting.",
    )
    parser.add_argument(
        "--max-nonfinite-skips",
        type=int,
        default=100,
        help="Stop training after this many skipped nonfinite steps; <=0 disables the limit.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=1)
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
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection for debugging only.",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="simamba-pretrain")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-id", default=None, help="Optional W&B run ID to resume explicitly.")
    parser.add_argument(
        "--wandb-resume",
        choices=["allow", "never", "must", "auto"],
        default=None,
        help="Optional W&B resume policy when --wandb-id is set or recovered from a checkpoint.",
    )
    parser.add_argument(
        "--ignore-resume-wandb",
        action="store_true",
        help="Load model/optimizer state from --resume but start a fresh W&B run.",
    )
    parser.add_argument(
        "--wandb-console",
        choices=["auto", "off", "wrap", "redirect", "wrap_raw", "wrap_emu"],
        default=os.environ.get("WANDB_CONSOLE", "auto"),
        help="W&B console capture mode for stdout/stderr.",
    )

    parser.add_argument("--lm-eval-every", type=int, default=0, help="Run lm-eval every N steps. 0 disables.")
    parser.add_argument("--lm-eval-tasks", default="lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa")
    parser.add_argument("--lm-eval-batch-size", type=int, default=64)
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requested but CUDA is not available. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()}. "
            "This usually indicates a broken CUDA runtime in the active environment or a stale GPU allocation."
        )
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

    def max_start(self, seq_len: int) -> int:
        max_start = self.size - seq_len - 1
        if max_start < 0:
            raise ValueError(f"Dataset {self.size} too small for seq_len={seq_len}.")
        return max_start

    def batch_from_starts(self, starts, seq_len: int, device: torch.device):
        x = torch.empty((len(starts), seq_len), dtype=torch.long)
        y = torch.empty((len(starts), seq_len), dtype=torch.long)
        for i, start in enumerate(starts):
            start = int(start)
            chunk = np.asarray(self.tokens[start : start + seq_len + 1], dtype=np.int64)
            x[i] = torch.from_numpy(chunk[:-1].copy())
            y[i] = torch.from_numpy(chunk[1:].copy())
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device):
        starts = torch.randint(0, self.max_start(seq_len) + 1, (batch_size,))
        return self.batch_from_starts(starts.tolist(), seq_len, device)


class EpochBlockSampler:
    def __init__(
        self,
        dataset: PackedTokenDataset,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        rank: int,
        world_size: int,
        seed: int,
        start_step: int,
        grad_accum_steps: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.global_batch_size = self.batch_size * self.world_size
        self.num_blocks = self.dataset.max_start(self.seq_len) // self.seq_len + 1
        if self.num_blocks <= 0:
            raise ValueError(f"Dataset {self.dataset.size} too small for seq_len={self.seq_len}.")
        self.blocks_consumed = (
            int(start_step) * int(grad_accum_steps) * self.global_batch_size
        )
        self._cached_epoch = None
        self._cached_permutation = None

    def _permutation_for_epoch(self, epoch: int):
        if self._cached_epoch != epoch:
            rng = np.random.default_rng(self.seed + int(epoch))
            self._cached_permutation = rng.permutation(self.num_blocks)
            self._cached_epoch = int(epoch)
        return self._cached_permutation

    def sample_batch(self):
        rank_offset = self.rank * self.batch_size
        starts = []
        for local_idx in range(self.batch_size):
            global_position = self.blocks_consumed + rank_offset + local_idx
            epoch, block_offset = divmod(global_position, self.num_blocks)
            block_id = int(self._permutation_for_epoch(epoch)[block_offset])
            starts.append(block_id * self.seq_len)
        self.blocks_consumed += self.global_batch_size
        return self.dataset.batch_from_starts(starts, self.seq_len, self.device)

    @property
    def epoch(self) -> int:
        return self.blocks_consumed // self.num_blocks

    @property
    def epoch_progress(self) -> float:
        return (self.blocks_consumed % self.num_blocks) / self.num_blocks


class FixedValidationSampler:
    def __init__(
        self,
        dataset: PackedTokenDataset,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        eval_iters: int,
        rank: int,
        world_size: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.eval_iters = int(eval_iters)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.global_batch_size = self.batch_size * self.world_size
        max_start = self.dataset.max_start(self.seq_len)
        rng = np.random.default_rng(self.seed)
        global_starts = rng.integers(
            0,
            max_start + 1,
            size=(self.eval_iters, self.global_batch_size),
            dtype=np.int64,
        )
        rank_slice = slice(self.rank * self.batch_size, (self.rank + 1) * self.batch_size)
        self.starts = np.ascontiguousarray(global_starts[:, rank_slice])

    def sample_batch(self, eval_iter: int):
        return self.dataset.batch_from_starts(self.starts[int(eval_iter)], self.seq_len, self.device)


def validate_simamba_correction_anneal_args(args):
    min_scale = float(args.simamba_correction_anneal_min)
    max_scale = float(args.simamba_correction_anneal_max)
    if not math.isfinite(min_scale) or not math.isfinite(max_scale):
        raise ValueError("Simamba correction anneal scales must be finite.")
    if min_scale < 0.0 or max_scale > 1.0 or min_scale > max_scale:
        raise ValueError(
            "Expected 0 <= --simamba-correction-anneal-min <= "
            "--simamba-correction-anneal-max <= 1."
        )
    if args.simamba_correction_anneal_start < 0:
        raise ValueError("--simamba-correction-anneal-start must be non-negative.")
    if args.simamba_correction_anneal_steps < 0:
        raise ValueError("--simamba-correction-anneal-steps must be non-negative.")


def simamba_correction_scale_for_step(step: int, args) -> float:
    if args.model_layer != "Simamba" or args.simamba_discretization != "simpson":
        return 1.0
    min_scale = float(args.simamba_correction_anneal_min)
    max_scale = float(args.simamba_correction_anneal_max)
    anneal_steps = int(args.simamba_correction_anneal_steps)
    start = int(args.simamba_correction_anneal_start)
    if anneal_steps <= 0:
        return max_scale
    if step <= start:
        progress = 0.0
    else:
        progress = min(1.0, (float(step) - float(start)) / float(anneal_steps))
    if args.simamba_correction_anneal_schedule == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return min_scale + (max_scale - min_scale) * progress


def _raw_model_for_helpers(model):
    raw_model = model.module if isinstance(model, DDP) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def set_simamba_correction_scale(model, scale: float) -> int:
    updated = 0
    for module in _raw_model_for_helpers(model).modules():
        setter = getattr(module, "set_simpson_correction_scale", None)
        if setter is not None:
            setter(scale)
            updated += 1
    return updated


def get_simamba_correction_scale(model) -> Optional[float]:
    for module in _raw_model_for_helpers(model).modules():
        getter = getattr(module, "get_simpson_correction_scale", None)
        if getter is not None:
            return float(getter())
    return None


def sync_simamba_correction_scale_to_config(model) -> None:
    raw_model = _raw_model_for_helpers(model)
    config = getattr(raw_model, "config", None)
    if config is None:
        return
    ssm_cfg = getattr(config, "ssm_cfg", None)
    if not isinstance(ssm_cfg, dict) or ssm_cfg.get("layer") != "Simamba":
        return
    scale = get_simamba_correction_scale(model)
    if scale is not None:
        ssm_cfg["simpson_correction_scale"] = scale


def make_config(args):
    if args.model_layer == "Simamba":
        ssm_cfg = {
            "layer": "Simamba",
            "d_state": args.simamba_d_state,
            "expand": args.simamba_expand,
            "headdim": args.simamba_headdim,
            "rope_fraction": args.simamba_rope_fraction,
            "chunk_size": args.simamba_chunk_size,
            "recompute_chunk_size": args.simamba_recompute_chunk_size,
            "dt_limit": tuple(args.simamba_dt_limit),
            "A_max": args.simamba_a_max,
            "d_conv": args.simamba_d_conv,
            "use_midpoint_control": args.simamba_use_midpoint_control,
            "control_logit_offset": args.simamba_control_logit_offset,
            "midpoint_logit_offset": args.simamba_midpoint_logit_offset,
            "simpson_correction_scale": simamba_correction_scale_for_step(0, args),
            "discretization": args.simamba_discretization,
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


def get_param_dtype(_dtype_name: str):
    # Keep master weights in fp32 and rely on autocast for reduced-precision activations.
    return torch.float32


def get_amp_dtype(dtype_name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]


def cuda_device_supports_bf16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    return props.major >= 8


def resolve_dtype_name(dtype_name: str, device: torch.device) -> str:
    if dtype_name == "auto":
        if device.type != "cuda":
            return "fp32"
        return "bf16" if cuda_device_supports_bf16(device) else "fp16"
    if dtype_name == "bf16" and device.type == "cuda" and not cuda_device_supports_bf16(device):
        props = torch.cuda.get_device_properties(device)
        raise ValueError(
            f"--dtype bf16 was requested, but {props.name} "
            f"(compute capability {props.major}.{props.minor}) does not support CUDA bf16. "
            "Use --dtype auto or --dtype fp16 on V100-class GPUs."
        )
    return dtype_name


def resolve_simamba_a_max(args, device: torch.device) -> tuple[float, Optional[float], Optional[float]]:
    requested_a_max = args.simamba_a_max
    if args.model_layer != "Simamba":
        return float(requested_a_max) if requested_a_max is not None else 16.0, requested_a_max, None

    fp16_triton_cuda = (
        args.dtype == "fp16"
        and args.simamba_backend in {"triton", "improved"}
        and device.type == "cuda"
    )
    if requested_a_max is None:
        effective_a_max = 4.0 if fp16_triton_cuda else 16.0
    else:
        effective_a_max = float(requested_a_max)

    stability_product = None
    if fp16_triton_cuda:
        dt_max = float(args.simamba_dt_limit[1])
        stability_product = effective_a_max * dt_max * int(args.simamba_chunk_size)
    return effective_a_max, requested_a_max, stability_product


def cast_optimizer_state_(optimizer):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if torch.is_floating_point(value) and key != "step":
                state[key] = value.to(device=param.device, dtype=param.dtype)
            else:
                state[key] = value.to(device=param.device)


def grad_health(model, *, max_names: int = 20):
    raw_model = model.module if isinstance(model, DDP) else model
    missing = []
    nonfinite = []
    for name, parameter in raw_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    return {
        "missing_count": len(missing),
        "missing_names": missing[:max_names],
        "nonfinite_count": len(nonfinite),
        "nonfinite_names": nonfinite[:max_names],
    }


def lr_for_step(step: int, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def build_wandb_metadata(wandb_run, args):
    if wandb_run is None:
        return None
    return {
        "id": wandb_run.id,
        "name": wandb_run.name,
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "group": getattr(wandb_run, "group", None) or args.wandb_group,
        "url": getattr(wandb_run, "url", None),
    }


def write_wandb_resume_metadata(path: Path, wandb_metadata):
    if wandb_metadata is None:
        return
    (path / "wandb_run.json").write_text(json.dumps(wandb_metadata, indent=2))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    resume_step: int,
    args,
    include_optimizer: bool,
    *,
    completed_step: Optional[int] = None,
    wandb_metadata=None,
):
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": raw_model.state_dict(),
        "step": resume_step,
        "resume_step": resume_step,
        "completed_step": completed_step,
        "args": vars(args),
        "config": asdict(raw_model.config),
    }
    if wandb_metadata is not None:
        payload["wandb"] = wandb_metadata
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def write_checkpoint_manifest(
    path: Path,
    *,
    step: int,
    kind: str,
    include_optimizer: bool,
    resume_step: Optional[int] = None,
    wandb_id: Optional[str] = None,
):
    payload = {
        "step": step,
        "kind": kind,
        "include_optimizer": include_optimizer,
        "created_at_unix": int(time.time()),
    }
    if resume_step is not None:
        payload["resume_step"] = resume_step
    if wandb_id is not None:
        payload["wandb_id"] = wandb_id
    (path / "checkpoint_manifest.json").write_text(json.dumps(payload, indent=2))


def load_checkpoint(path: Path, model, optimizer=None):
    # trainer.pt is a trusted local checkpoint that includes optimizer state,
    # Paths, and other non-tensor metadata, so weights_only=False is required
    # on PyTorch 2.6+.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "resume_step" in ckpt:
        resume_step = int(ckpt["resume_step"])
        completed_step = ckpt.get("completed_step")
    else:
        # Backward compatibility for checkpoints written before resume_step was
        # introduced. Those files stored the last completed step as "step".
        completed_step = ckpt.get("step")
        resume_step = (int(completed_step) + 1) if completed_step is not None else 0
    return {
        "resume_step": resume_step,
        "completed_step": completed_step,
        "wandb": ckpt.get("wandb"),
    }


@torch.no_grad()
def evaluate(model, dataset, args, device, amp_ctx, fixed_sampler: Optional[FixedValidationSampler] = None):
    model.eval()
    losses = []
    for eval_iter in range(args.eval_iters):
        if fixed_sampler is not None:
            x, y = fixed_sampler.sample_batch(eval_iter)
        else:
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


def save_training_snapshot(
    checkpoint_dir: Path,
    *,
    model,
    optimizer,
    resume_step: int,
    completed_step: Optional[int],
    args,
    include_optimizer: bool,
    kind: str,
    wandb_metadata,
):
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sync_simamba_correction_scale_to_config(model)
    save_checkpoint(
        checkpoint_dir / "trainer.pt",
        model,
        optimizer,
        resume_step,
        args,
        include_optimizer=include_optimizer,
        completed_step=completed_step,
        wandb_metadata=wandb_metadata,
    )
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.save_pretrained(checkpoint_dir)
    write_checkpoint_manifest(
        checkpoint_dir,
        step=resume_step,
        kind=kind,
        include_optimizer=include_optimizer,
        resume_step=resume_step,
        wandb_id=(wandb_metadata or {}).get("id"),
    )
    write_wandb_resume_metadata(checkpoint_dir, wandb_metadata)


def main():
    process_start = time.perf_counter()
    args = parse_args()
    validate_simamba_correction_anneal_args(args)
    if args.model_layer == "Simamba" and args.simamba_discretization == "trapezoid":
        if args.simamba_backend != "reference":
            raise ValueError(
                "--simamba-discretization trapezoid currently requires --simamba-backend reference."
            )
        if args.simamba_use_midpoint_control:
            raise ValueError("--simamba-use-midpoint-control is only valid with Simpson discretization.")
    emit_status(
        "startup.args_parsed",
        distributed_env=("RANK" in os.environ),
        train_data=str(args.train_data),
        val_data=(str(args.val_data) if args.val_data is not None else None),
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_console=args.wandb_console,
    )

    phase_start = time.perf_counter()
    emit_status("distributed.init.begin")
    distributed, rank, world_size, local_rank = setup_distributed()
    emit_status(
        "distributed.init.end",
        rank=rank,
        duration_sec=round(time.perf_counter() - phase_start, 3),
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    requested_dtype = args.dtype
    args.dtype = resolve_dtype_name(args.dtype, device)
    (
        args.simamba_a_max,
        requested_simamba_a_max,
        simamba_fp16_stability_product,
    ) = resolve_simamba_a_max(args, device)
    emit_status(
        "runtime.device_ready",
        rank=rank,
        device=str(device),
        requested_dtype=requested_dtype,
        dtype=args.dtype,
        param_dtype=str(get_param_dtype(args.dtype)),
        cuda_bf16_supported=(cuda_device_supports_bf16(device) if device.type == "cuda" else None),
    )
    if args.model_layer == "Simamba":
        emit_status(
            "runtime.simamba_dynamics_ready",
            rank=rank,
            requested_a_max=requested_simamba_a_max,
            effective_a_max=args.simamba_a_max,
            dt_limit=list(args.simamba_dt_limit),
            chunk_size=args.simamba_chunk_size,
            discretization=args.simamba_discretization,
            backend=args.simamba_backend,
            correction_anneal_min=args.simamba_correction_anneal_min,
            correction_anneal_max=args.simamba_correction_anneal_max,
            correction_anneal_start=args.simamba_correction_anneal_start,
            correction_anneal_steps=args.simamba_correction_anneal_steps,
            correction_anneal_schedule=args.simamba_correction_anneal_schedule,
            fp16_stability_product=simamba_fp16_stability_product,
        )
        if (
            args.dtype == "fp16"
            and args.simamba_backend in {"triton", "improved"}
            and simamba_fp16_stability_product is not None
            and simamba_fp16_stability_product > 8.0
        ):
            emit_status(
                "runtime.simamba_fp16_stability_warning",
                rank=rank,
                effective_a_max=args.simamba_a_max,
                dt_max=float(args.simamba_dt_limit[1]),
                chunk_size=args.simamba_chunk_size,
                product=simamba_fp16_stability_product,
            )

    phase_start = time.perf_counter()
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
    emit_status("runtime.seeded", rank=rank, duration_sec=round(time.perf_counter() - phase_start, 3), seed=args.seed + rank)

    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    emit_status("runtime.batch_shape_ready", rank=rank, grad_accum_steps=grad_accum_steps)

    phase_start = time.perf_counter()
    emit_status("dataset.open.begin", rank=rank)
    train_data = PackedTokenDataset(args.train_data, args.token_dtype)
    val_data = PackedTokenDataset(args.val_data, args.token_dtype) if args.val_data is not None else None
    emit_status(
        "dataset.open.end",
        rank=rank,
        duration_sec=round(time.perf_counter() - phase_start, 3),
        train_tokens=train_data.size,
        val_tokens=(val_data.size if val_data is not None else None),
    )

    phase_start = time.perf_counter()
    emit_status("model.init.begin", rank=rank, model_layer=args.model_layer)
    config = make_config(args)
    model = MambaLMHeadModel(config=config, device=device, dtype=get_param_dtype(args.dtype))
    emit_status(
        "model.init.end",
        rank=rank,
        duration_sec=round(time.perf_counter() - phase_start, 3),
        param_count=count_parameters(model),
    )
    if args.compile:
        compile_start = time.perf_counter()
        emit_status("model.compile.begin", rank=rank)
        model = torch.compile(model)
        emit_status("model.compile.end", rank=rank, duration_sec=round(time.perf_counter() - compile_start, 3))
    model.train()

    phase_start = time.perf_counter()
    emit_status("optimizer.init.begin", rank=rank)
    optimizer, optimizer_stats = build_optimizer(model, args)
    emit_status(
        "optimizer.init.end",
        rank=rank,
        duration_sec=round(time.perf_counter() - phase_start, 3),
        decay_tensors=optimizer_stats["decay_tensors"],
        no_decay_tensors=optimizer_stats["no_decay_tensors"],
        decay_elements=optimizer_stats["decay_elements"],
        no_decay_elements=optimizer_stats["no_decay_elements"],
        weight_decay=args.weight_decay,
    )

    start_step = 0
    resume_wandb = None
    if args.resume is not None:
        phase_start = time.perf_counter()
        emit_status("checkpoint.resume.begin", rank=rank, path=str(args.resume))
        resume_state = load_checkpoint(args.resume, model, optimizer)
        cast_optimizer_state_(optimizer)
        start_step = resume_state["resume_step"]
        resume_wandb = resume_state.get("wandb")
        if args.ignore_resume_wandb:
            resume_wandb = None
        emit_status(
            "checkpoint.resume.end",
            rank=rank,
            duration_sec=round(time.perf_counter() - phase_start, 3),
            resumed_step=start_step,
        )
    else:
        emit_status("checkpoint.resume.skipped", rank=rank)

    if distributed:
        phase_start = time.perf_counter()
        emit_status("ddp.wrap.begin", rank=rank)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
        emit_status(
            "ddp.wrap.end",
            rank=rank,
            duration_sec=round(time.perf_counter() - phase_start, 3),
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    simamba_layer_count = 0
    initial_simamba_correction_scale = simamba_correction_scale_for_step(start_step, args)
    if args.model_layer == "Simamba":
        simamba_layer_count = set_simamba_correction_scale(model, initial_simamba_correction_scale)
        emit_status(
            "runtime.simamba_correction_scale_ready",
            rank=rank,
            start_step=start_step,
            scale=initial_simamba_correction_scale,
            layer_count=simamba_layer_count,
        )

    args.eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 100000
    train_sampler = None
    if args.train_sampling == "epoch":
        train_sampler = EpochBlockSampler(
            train_data,
            batch_size=args.micro_batch_size,
            seq_len=args.seq_len,
            device=device,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            start_step=start_step,
            grad_accum_steps=grad_accum_steps,
        )

    fixed_val_sampler = None
    if val_data is not None and args.eval_sampling == "fixed":
        fixed_val_sampler = FixedValidationSampler(
            val_data,
            batch_size=args.micro_batch_size,
            seq_len=args.seq_len,
            device=device,
            eval_iters=args.eval_iters,
            rank=rank,
            world_size=world_size,
            seed=args.eval_seed,
        )

    emit_status(
        "dataset.sampling_ready",
        rank=rank,
        train_sampling=args.train_sampling,
        train_blocks=(train_sampler.num_blocks if train_sampler is not None else None),
        train_sampler_epoch=(int(train_sampler.epoch) if train_sampler is not None else None),
        train_sampler_epoch_progress=(
            round(float(train_sampler.epoch_progress), 6) if train_sampler is not None else None
        ),
        eval_sampling=(args.eval_sampling if val_data is not None else None),
        eval_seed=(args.eval_seed if fixed_val_sampler is not None else None),
        fixed_eval_batches=(args.eval_iters if fixed_val_sampler is not None else None),
        fixed_eval_global_examples=(
            args.eval_iters * args.micro_batch_size * world_size
            if fixed_val_sampler is not None
            else None
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.dtype == "fp16" and device.type == "cuda"),
        init_scale=args.grad_scale_init,
        growth_interval=args.grad_scale_growth_interval,
    )
    amp_dtype = get_amp_dtype(args.dtype)
    if args.dtype != "fp32":
        amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    wandb_run = None
    if args.wandb and is_master(rank):
        phase_start = time.perf_counter()
        emit_status("wandb.init.begin", rank=rank)
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "W&B logging was requested, but 'wandb' is not installed in the active interpreter. "
                f"Install it with: {sys.executable} -m pip install wandb "
                "or sync the optional 'train' dependencies declared in pyproject.toml."
            ) from exc

        effective_wandb = resume_wandb or {}
        wandb_id = args.wandb_id or effective_wandb.get("id")
        wandb_resume = args.wandb_resume or ("allow" if wandb_id is not None else None)
        wandb_name = effective_wandb.get("name") if wandb_id is not None else args.wandb_name
        wandb_group = effective_wandb.get("group") if wandb_id is not None else args.wandb_group
        wandb_project = effective_wandb.get("project") if wandb_id is not None else args.wandb_project
        wandb_entity = effective_wandb.get("entity") if wandb_id is not None else args.wandb_entity
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=wandb_id,
            resume=wandb_resume,
            name=wandb_name,
            group=wandb_group,
            config=vars(args),
            settings=wandb.Settings(console=args.wandb_console),
        )
        emit_status(
            "wandb.init.end",
            rank=rank,
            duration_sec=round(time.perf_counter() - phase_start, 3),
            run_id=wandb_run.id,
            run_name=wandb_run.name,
            run_url=wandb_run.url,
        )
    elif is_master(rank):
        emit_status("wandb.init.skipped", rank=rank)

    tokens_per_step = args.global_batch_size * args.seq_len
    last_log_time = time.time()
    best_val_loss: Optional[float] = None
    nonfinite_skip_count = 0
    amp_overflow_skip_count = 0
    emit_status(
        "train.loop.begin",
        rank=rank,
        startup_sec=round(time.perf_counter() - process_start, 3),
        start_step=start_step,
        max_steps=args.max_steps,
        tokens_per_step=tokens_per_step,
    )

    stop_requested = False
    stop_signal = None

    def _request_stop(signum, _frame):
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = signal.Signals(signum).name
        emit_status("runtime.stop_requested", rank=rank, signal=stop_signal)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    next_step_to_run = start_step

    for step in range(start_step, args.max_steps):
        if stop_requested:
            break
        if is_master(rank) and step == start_step:
            step_start = time.perf_counter()
            emit_status("train.first_step.begin", rank=rank, step=step)
        lr = lr_for_step(step, args)
        simamba_correction_scale = simamba_correction_scale_for_step(step, args)
        if simamba_layer_count > 0:
            set_simamba_correction_scale(model, simamba_correction_scale)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if is_master(rank) and step == start_step:
                micro_step_start = time.perf_counter()
                emit_status(
                    "train.first_step.microbatch.begin",
                    rank=rank,
                    step=step,
                    micro_step=micro_step,
                    grad_accum_steps=grad_accum_steps,
                )
            sync_context = model.no_sync() if distributed and micro_step + 1 < grad_accum_steps else nullcontext()
            with sync_context:
                if train_sampler is not None:
                    x, y = train_sampler.sample_batch()
                else:
                    x, y = train_data.sample_batch(args.micro_batch_size, args.seq_len, device)
                with amp_ctx():
                    logits = model(x).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward()
            if is_master(rank) and step == start_step:
                emit_status(
                    "train.first_step.microbatch.end",
                    rank=rank,
                    step=step,
                    micro_step=micro_step,
                    duration_sec=round(time.perf_counter() - micro_step_start, 3),
                    loss=float(loss.detach().item() * grad_accum_steps),
                )

        health = grad_health(model)
        health_missing = torch.tensor(health["missing_count"], device=device, dtype=torch.int32)
        health_nonfinite = torch.tensor(health["nonfinite_count"], device=device, dtype=torch.int32)
        if distributed:
            dist.all_reduce(health_missing, op=dist.ReduceOp.SUM)
            dist.all_reduce(health_nonfinite, op=dist.ReduceOp.SUM)
        if int(health_missing.item()) > 0 or int(health_nonfinite.item()) > 0:
            emit_status(
                "train.grad_health",
                rank=rank,
                step=step,
                local_missing_count=health["missing_count"],
                local_missing_names=health["missing_names"],
                local_nonfinite_count=health["nonfinite_count"],
                local_nonfinite_names=health["nonfinite_names"],
                global_missing_count=int(health_missing.item()),
                global_nonfinite_count=int(health_nonfinite.item()),
            )

        grad_norm_post_clip = torch.zeros((), device=device)
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.zeros((), device=device)

        loss_is_finite = bool(torch.isfinite(loss_accum).item())
        grad_norm_is_finite = bool(torch.isfinite(grad_norm).item())
        loss_finite_step = torch.tensor(1 if loss_is_finite else 0, device=device, dtype=torch.int32)
        grad_norm_finite_step = torch.tensor(1 if grad_norm_is_finite else 0, device=device, dtype=torch.int32)
        if distributed:
            dist.all_reduce(loss_finite_step, op=dist.ReduceOp.MIN)
            dist.all_reduce(grad_norm_finite_step, op=dist.ReduceOp.MIN)
        global_loss_is_finite = bool(int(loss_finite_step.item()))
        global_grad_norm_is_finite = bool(int(grad_norm_finite_step.item()))
        if not (global_loss_is_finite and global_grad_norm_is_finite):
            optimizer.zero_grad(set_to_none=True)
            if global_loss_is_finite and scaler.is_enabled():
                amp_overflow_skip_count += 1
                scaler.update()
                next_step_to_run = step + 1
                emit_status(
                    "train.amp_overflow_skipped",
                    rank=rank,
                    step=step,
                    lr=lr,
                    grad_norm_finite=grad_norm_is_finite,
                    grad_norm=(float(grad_norm.item()) if grad_norm_is_finite else None),
                    amp_overflow_skip_count=amp_overflow_skip_count,
                    grad_scale=float(scaler.get_scale()),
                )
                if is_master(rank):
                    metrics = {
                        "step": step,
                        "train/amp_overflow_skip": 1,
                        "train/amp_overflow_skip_count": amp_overflow_skip_count,
                        "train/lr": lr,
                        "train/grad_scale": float(scaler.get_scale()),
                    }
                    print(json.dumps(metrics), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                continue
            if args.skip_nonfinite_steps:
                nonfinite_skip_count += 1
                if scaler.is_enabled():
                    scaler.update()
                next_step_to_run = step + 1
                emit_status(
                    "train.nonfinite_skipped",
                    rank=rank,
                    step=step,
                    lr=lr,
                    loss_finite=loss_is_finite,
                    grad_norm_finite=grad_norm_is_finite,
                    global_loss_finite=global_loss_is_finite,
                    global_grad_norm_finite=global_grad_norm_is_finite,
                    loss=(float(loss_accum.item()) if loss_is_finite else None),
                    grad_norm=(float(grad_norm.item()) if grad_norm_is_finite else None),
                    nonfinite_skip_count=nonfinite_skip_count,
                    max_nonfinite_skips=args.max_nonfinite_skips,
                    grad_scale=(float(scaler.get_scale()) if scaler.is_enabled() else None),
                )
                if is_master(rank):
                    metrics = {
                        "step": step,
                        "train/nonfinite_skip": 1,
                        "train/nonfinite_skip_count": nonfinite_skip_count,
                        "train/lr": lr,
                    }
                    print(json.dumps(metrics), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                if args.max_nonfinite_skips > 0 and nonfinite_skip_count >= args.max_nonfinite_skips:
                    stop_requested = True
                    stop_signal = "NONFINITE_SKIP_LIMIT"
                    emit_status(
                        "train.nonfinite_skip_limit",
                        rank=rank,
                        step=step,
                        nonfinite_skip_count=nonfinite_skip_count,
                    )
                    break
                continue
            stop_requested = True
            stop_signal = "NONFINITE"
            emit_status(
                "train.nonfinite_detected",
                rank=rank,
                step=step,
                lr=lr,
                loss_finite=loss_is_finite,
                grad_norm_finite=grad_norm_is_finite,
                loss=(float(loss_accum.item()) if loss_is_finite else None),
                grad_norm=(float(grad_norm.item()) if grad_norm_is_finite else None),
            )
            break

        grad_norm_post_clip = (
            grad_norm.clamp(max=args.grad_clip) if args.grad_clip > 0 else grad_norm
        )
        grad_clip_coef = (
            torch.clamp(args.grad_clip / grad_norm.clamp(min=1e-12), max=1.0)
            if args.grad_clip > 0
            else torch.ones((), device=device)
        )

        scaler.step(optimizer)
        scaler.update()

        if distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        next_step_to_run = step + 1

        if is_master(rank) and step == start_step:
            emit_status(
                "train.first_step.end",
                rank=rank,
                step=step,
                duration_sec=round(time.perf_counter() - step_start, 3),
                loss=float(loss_accum.item()),
            )

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
                "train/grad_norm_pre_clip": float(grad_norm.item()),
                "train/grad_norm_post_clip": float(grad_norm_post_clip.item()),
                "train/grad_clip_coef": float(grad_clip_coef.item()),
                "train/tokens_per_sec": tokens_per_sec,
            }
            if simamba_layer_count > 0:
                metrics["train/simpson_correction_scale"] = simamba_correction_scale
            if train_sampler is not None:
                metrics["train/sampler_epoch"] = int(train_sampler.epoch)
                metrics["train/sampler_epoch_progress"] = float(train_sampler.epoch_progress)
            print(json.dumps(metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        if val_data is not None and step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args, device, amp_ctx, fixed_val_sampler)
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
                    wandb_metadata = build_wandb_metadata(wandb_run, args)
                    save_training_snapshot(
                        tmp_dir,
                        model=model,
                        optimizer=optimizer,
                        resume_step=step + 1,
                        completed_step=step,
                        args=args,
                        include_optimizer=False,
                        kind="best",
                        wandb_metadata=wandb_metadata,
                    )
                    (tmp_dir / "metrics.json").write_text(json.dumps({"step": step, "val/loss": best_val_loss}, indent=2))
                    replace_dir(tmp_dir, best_dir)
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        if is_master(rank) and step > 0 and step % args.save_every == 0:
            latest_dir = args.output_dir / "latest"
            tmp_latest = args.output_dir / ".latest_build"
            if tmp_latest.exists():
                shutil.rmtree(tmp_latest)
            wandb_metadata = build_wandb_metadata(wandb_run, args)
            save_training_snapshot(
                tmp_latest,
                model=model,
                optimizer=optimizer,
                resume_step=step + 1,
                completed_step=step,
                args=args,
                include_optimizer=args.save_optimizer_latest_only,
                kind="latest",
                wandb_metadata=wandb_metadata,
            )
            replace_dir(tmp_latest, latest_dir)
            shutil.rmtree(tmp_latest, ignore_errors=True)
            write_wandb_resume_metadata(args.output_dir, wandb_metadata)

            milestone_dir = args.output_dir / f"step_{step:07d}"
            if args.keep_milestones > 0:
                if milestone_dir.exists():
                    shutil.rmtree(milestone_dir)
                shutil.copytree(latest_dir, milestone_dir)
                trainer_path = milestone_dir / "trainer.pt"
                if trainer_path.exists() and args.save_optimizer_latest_only:
                    ckpt = torch.load(trainer_path, map_location="cpu", weights_only=False)
                    ckpt.pop("optimizer", None)
                    torch.save(ckpt, trainer_path)
                write_checkpoint_manifest(
                    milestone_dir,
                    step=step + 1,
                    kind="milestone",
                    include_optimizer=False,
                    resume_step=step + 1,
                    wandb_id=(wandb_metadata or {}).get("id"),
                )
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

        if stop_requested:
            break

    completed_all_steps = next_step_to_run >= args.max_steps
    if is_master(rank):
        phase_start = time.perf_counter()
        checkpoint_step = args.max_steps if completed_all_steps else next_step_to_run
        emit_status(
            "checkpoint.final.begin",
            rank=rank,
            step=checkpoint_step,
            interrupted=(not completed_all_steps),
            signal=stop_signal,
        )
        ckpt_dir = args.output_dir / "latest"
        tmp_dir = args.output_dir / ".final_build"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        wandb_metadata = build_wandb_metadata(wandb_run, args)
        save_training_snapshot(
            tmp_dir,
            model=model,
            optimizer=optimizer,
            resume_step=checkpoint_step,
            completed_step=(checkpoint_step - 1 if checkpoint_step > 0 else None),
            args=args,
            include_optimizer=args.save_optimizer_latest_only,
            kind="latest",
            wandb_metadata=wandb_metadata,
        )
        replace_dir(tmp_dir, ckpt_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        write_wandb_resume_metadata(args.output_dir, wandb_metadata)
        emit_status(
            "checkpoint.final.end",
            rank=rank,
            duration_sec=round(time.perf_counter() - phase_start, 3),
            step=checkpoint_step,
            interrupted=(not completed_all_steps),
            signal=stop_signal,
        )

    if wandb_run is not None:
        phase_start = time.perf_counter()
        emit_status("wandb.finish.begin", rank=rank)
        wandb_run.finish()
        emit_status("wandb.finish.end", rank=rank, duration_sec=round(time.perf_counter() - phase_start, 3))

    if distributed:
        emit_status("distributed.shutdown.begin", rank=rank)
        dist.barrier()
        dist.destroy_process_group()
        emit_status("distributed.shutdown.end", rank=rank)


if __name__ == "__main__":
    main()
