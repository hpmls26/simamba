#!/usr/bin/env python

"""Benchmark Simamba (Simpson) vs Mamba-3 (trapezoidal) SISO kernels.

This harness follows repository benchmark style:
- CLI via argparse
- CUDA synchronized wall-clock timing for end-to-end runs
- Triton do_bench median timing for microbenchmarks

Examples:
    python benchmarks/benchmark_simamba_siso.py --mode all
    python benchmarks/benchmark_simamba_siso.py --mode micro --batch 8 --prompt-len 2048
    python benchmarks/benchmark_simamba_siso.py --mode e2e --batch 1 --prompt-len 512 --gen-len 256
"""

import argparse
import gc
import sys
import time
import types
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

import torch


def _install_import_shims() -> None:
    # Keep this benchmark runnable without requiring full extension install.
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

    if "mamba_ssm" not in sys.modules:
        repo_root = Path(__file__).resolve().parents[1]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(repo_root / "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg


_install_import_shims()

from triton.testing import do_bench

try:
    from triton.testing import do_bench_cudagraph
except Exception:
    do_bench_cudagraph = None

from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd as mamba3_trap_siso_fwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step
from mamba_ssm.ops.triton.simamba.mamba3_siso_fwd import mamba3_siso_fwd as simamba_siso_fwd
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step as simamba_siso_step


T = TypeVar("T")


def _dtype_from_name(name: str) -> torch.dtype:
    key = name.lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if key == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _cuda_preflight(device: str, retries: int, sleep_s: float) -> None:
    """Initialize a CUDA context with optional retry for transient busy devices."""
    attempts = max(1, retries)
    last_error: Optional[Exception] = None
    base_device = torch.device(device)
    if base_device.type != "cuda":
        raise SystemExit(f"Expected CUDA device, got: {device}")

    for attempt in range(1, attempts + 1):
        try:
            dev_index = base_device.index if base_device.index is not None else torch.cuda.current_device()
            resolved_device = torch.device("cuda", dev_index)
            torch.cuda.set_device(resolved_device)
            probe = torch.empty(1, device=resolved_device, dtype=torch.float32)
            del probe
            torch.cuda.synchronize()
            return
        except Exception as exc:
            last_error = exc
            is_busy = "busy or unavailable" in str(exc).lower()
            if is_busy and attempt < attempts:
                torch.cuda.empty_cache()
                time.sleep(max(0.0, sleep_s))
                continue
            break

    hint = (
        "CUDA initialization failed before benchmark tensor allocation. "
        "Run in a GPU-allocated shell and verify the selected CUDA_VISIBLE_DEVICES "
        "points to a healthy, available GPU."
    )
    if last_error is not None:
        raise SystemExit(f"{hint}\nOriginal error: {last_error}") from last_error
    raise SystemExit(hint)


def _retry_cuda_busy(op_name: str, fn: Callable[[], T], retries: int, sleep_s: float) -> T:
    attempts = max(1, retries)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            is_busy = "busy or unavailable" in str(exc).lower()
            if is_busy and attempt < attempts:
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(max(0.0, sleep_s))
                continue
            break

    hint = f"CUDA failed during {op_name}."
    if last_error is not None:
        raise SystemExit(f"{hint}\nOriginal error: {last_error}") from last_error
    raise SystemExit(hint)


def _reset_peak_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _peak_memory_mb(fn, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _reset_peak_memory()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _bench_median_ms(fn, warmup: int, rep: int) -> float:
    return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def _bench_step_ms(fn, warmup: int, rep: int) -> float:
    if do_bench_cudagraph is None:
        return _bench_median_ms(fn, warmup=warmup, rep=rep)
    return float(do_bench_cudagraph(fn, rep=rep))


def _time_ms(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) * 1000.0 / repeats


def _angle_cumsum_and_final(
    angles: torch.Tensor,
    dt: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Matches the angle_dt preprocessing used by the Mamba-3 combined wrapper.
    increments = torch.tanh(angles.float()) * dt.transpose(-1, -2).unsqueeze(-1).float() * torch.pi
    cumsum = torch.cumsum(increments, dim=1)
    if init_state is not None:
        cumsum = cumsum + init_state.unsqueeze(1).float()
    cumsum = torch.remainder(cumsum, 2.0 * torch.pi)
    return cumsum, cumsum[:, -1]


def _mamba3_prefill(
    *,
    q,
    k,
    v,
    adt,
    dt,
    trap,
    q_bias,
    k_bias,
    angles,
    d,
    z,
    chunk_size: int,
    return_final_states: bool,
):
    angles_cumsum, final_angle_state = _angle_cumsum_and_final(angles, dt)
    out = mamba3_trap_siso_fwd(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Trap=trap,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles_cumsum,
        D=d,
        Z=z,
        Initial_States=None,
        chunk_size=chunk_size,
        store_states_adt_outv=False,
        return_final_states=return_final_states,
        cu_seqlens=None,
    )
    y = out[0]
    if not return_final_states:
        return y, None
    final_states = out[-1]
    if final_states is None:
        raise RuntimeError("Mamba-3 forward did not return final states.")
    final_ssm_state, final_k_state, final_v_state = final_states
    step_states = (final_angle_state, final_ssm_state, final_k_state, final_v_state)
    return y, step_states


def _make_sequence_tensors(
    *,
    batch: int,
    seqlen: int,
    nheads: int,
    nheads_qk: int,
    headdim_qk: int,
    headdim_v: int,
    n_angles: int,
    dtype: torch.dtype,
    use_z: bool,
    use_midpoint: bool,
    device: str,
):
    q = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=dtype)

    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    coeff = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    midpoint = (
        torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
        if use_midpoint
        else None
    )
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=dtype) if use_z else None
    return {
        "q": q,
        "k": k,
        "v": v,
        "adt": adt,
        "dt": dt,
        "coeff": coeff,
        "midpoint": midpoint,
        "angles": angles,
        "z": z,
    }


def _zero_trap_states(
    *,
    batch: int,
    nheads: int,
    headdim_angles: int,
    headdim_v: int,
    headdim_qk: int,
    v_dtype: torch.dtype,
    device: str,
):
    return (
        torch.zeros(batch, nheads, headdim_angles, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_qk, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_v, device=device, dtype=v_dtype),
    )


def _zero_sim_states(
    *,
    batch: int,
    nheads: int,
    headdim_angles: int,
    headdim_v: int,
    headdim_qk: int,
    qk_dtype: torch.dtype,
    v_dtype: torch.dtype,
    device: str,
):
    return (
        torch.zeros(batch, nheads, headdim_angles, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_qk, device=device, dtype=qk_dtype),
        torch.zeros(batch, nheads, headdim_qk, device=device, dtype=qk_dtype),
        torch.zeros(batch, nheads, headdim_v, device=device, dtype=v_dtype),
        torch.zeros(batch, nheads, headdim_v, device=device, dtype=v_dtype),
    )


def _delta_pct(mamba3_val: float, simamba_val: float) -> float:
    if mamba3_val == 0:
        return 0.0
    return (simamba_val / mamba3_val - 1.0) * 100.0


def _print_row(metric: str, m3: float, sim: float, unit: str) -> None:
    print(f"{metric:<20} {m3:>12.4f} {sim:>12.4f} {_delta_pct(m3, sim):>+9.2f}% {unit}")


def _run_micro(
    *,
    args,
    params,
    prefill,
    decode,
    nheads_qk: int,
    n_angles: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    def m3_prefill_fn():
        out, _ = _mamba3_prefill(
            q=prefill["q"],
            k=prefill["k"],
            v=prefill["v"],
            adt=prefill["adt"],
            dt=prefill["dt"],
            trap=prefill["coeff"],
            q_bias=params["q_bias"],
            k_bias=params["k_bias"],
            angles=prefill["angles"],
            d=params["d"],
            z=prefill["z"],
            chunk_size=args.chunk_size,
            return_final_states=False,
        )
        return out

    def sim_prefill_fn():
        return simamba_siso_fwd(
            Q=prefill["q"],
            K=prefill["k"],
            V=prefill["v"],
            ADT=prefill["adt"],
            DT=prefill["dt"],
            Simpson=prefill["coeff"],
            Midpoint=prefill["midpoint"],
            Q_bias=params["q_bias"],
            K_bias=params["k_bias"],
            Angles=prefill["angles"],
            D=params["d"],
            Z=prefill["z"],
            Initial_States=None,
            chunk_size=args.chunk_size,
            return_final_states=False,
        )[0]

    m3_prefill_fn()
    sim_prefill_fn()

    m3_prefill_ms = _bench_median_ms(m3_prefill_fn, warmup=args.warmup, rep=args.rep)
    sim_prefill_ms = _bench_median_ms(sim_prefill_fn, warmup=args.warmup, rep=args.rep)
    m3_prefill_mb = _peak_memory_mb(m3_prefill_fn, warmup=1)
    sim_prefill_mb = _peak_memory_mb(sim_prefill_fn, warmup=1)

    step_idx = 0
    m3_step_state = _zero_trap_states(
        batch=args.batch,
        nheads=args.nheads,
        headdim_angles=n_angles,
        headdim_v=args.headdim_v,
        headdim_qk=args.headdim_qk,
        v_dtype=dtype,
        device=device,
    )
    sim_step_state = _zero_sim_states(
        batch=args.batch,
        nheads=args.nheads,
        headdim_angles=n_angles,
        headdim_v=args.headdim_v,
        headdim_qk=args.headdim_qk,
        qk_dtype=dtype,
        v_dtype=dtype,
        device=device,
    )

    def m3_step_fn():
        out, _ = mamba3_siso_step(
            Q=decode["q"][:, step_idx],
            K=decode["k"][:, step_idx],
            V=decode["v"][:, step_idx],
            ADT=decode["adt"][:, :, step_idx],
            DT=decode["dt"][:, :, step_idx],
            Trap=decode["coeff"][:, :, step_idx],
            Q_bias=params["q_bias"],
            K_bias=params["k_bias"],
            Angles=decode["angles"][:, step_idx],
            D=params["d"],
            Z=decode["z"][:, step_idx] if decode["z"] is not None else None,
            Input_States=m3_step_state,
        )
        return out

    def sim_step_fn():
        out, _ = simamba_siso_step(
            Q=decode["q"][:, step_idx],
            K=decode["k"][:, step_idx],
            V=decode["v"][:, step_idx],
            ADT=decode["adt"][:, :, step_idx],
            DT=decode["dt"][:, :, step_idx],
            Simpson=decode["coeff"][:, :, step_idx],
            Midpoint=(decode["midpoint"][:, :, step_idx] if decode["midpoint"] is not None else None),
            Q_bias=params["q_bias"],
            K_bias=params["k_bias"],
            Angles=decode["angles"][:, step_idx],
            D=params["d"],
            Z=decode["z"][:, step_idx] if decode["z"] is not None else None,
            Input_States=sim_step_state,
        )
        return out

    m3_step_fn()
    sim_step_fn()

    m3_step_ms = _bench_step_ms(m3_step_fn, warmup=args.warmup, rep=args.step_rep)
    sim_step_ms = _bench_step_ms(sim_step_fn, warmup=args.warmup, rep=args.step_rep)
    m3_step_mb = _peak_memory_mb(m3_step_fn, warmup=1)
    sim_step_mb = _peak_memory_mb(sim_step_fn, warmup=1)

    prefill_tokens = args.batch * args.prompt_len
    m3_prefill_toks = prefill_tokens / (m3_prefill_ms * 1e-3)
    sim_prefill_toks = prefill_tokens / (sim_prefill_ms * 1e-3)

    step_tokens = args.batch
    m3_step_toks = step_tokens / (m3_step_ms * 1e-3)
    sim_step_toks = step_tokens / (sim_step_ms * 1e-3)

    print("\n[Micro] median runtime")
    print(f"{'metric':<20} {'mamba3':>12} {'simamba':>12} {'sim/m3':>9} unit")
    _print_row("prefill_ms", m3_prefill_ms, sim_prefill_ms, "ms")
    _print_row("prefill_toks_s", m3_prefill_toks, sim_prefill_toks, "tok/s")
    _print_row("step_ms", m3_step_ms, sim_step_ms, "ms")
    _print_row("step_toks_s", m3_step_toks, sim_step_toks, "tok/s")
    _print_row("prefill_peak_mb", m3_prefill_mb, sim_prefill_mb, "MB")
    _print_row("step_peak_mb", m3_step_mb, sim_step_mb, "MB")


def _run_e2e(*, args, params, prefill, decode) -> None:
    def m3_e2e_once():
        _, states = _mamba3_prefill(
            q=prefill["q"],
            k=prefill["k"],
            v=prefill["v"],
            adt=prefill["adt"],
            dt=prefill["dt"],
            trap=prefill["coeff"],
            q_bias=params["q_bias"],
            k_bias=params["k_bias"],
            angles=prefill["angles"],
            d=params["d"],
            z=prefill["z"],
            chunk_size=args.chunk_size,
            return_final_states=True,
        )
        if states is None:
            raise RuntimeError("Missing Mamba-3 prefill states for decode.")
        for t in range(args.gen_len):
            _, states = mamba3_siso_step(
                Q=decode["q"][:, t],
                K=decode["k"][:, t],
                V=decode["v"][:, t],
                ADT=decode["adt"][:, :, t],
                DT=decode["dt"][:, :, t],
                Trap=decode["coeff"][:, :, t],
                Q_bias=params["q_bias"],
                K_bias=params["k_bias"],
                Angles=decode["angles"][:, t],
                D=params["d"],
                Z=decode["z"][:, t] if decode["z"] is not None else None,
                Input_States=states,
            )

    def sim_e2e_once():
        out = simamba_siso_fwd(
            Q=prefill["q"],
            K=prefill["k"],
            V=prefill["v"],
            ADT=prefill["adt"],
            DT=prefill["dt"],
            Simpson=prefill["coeff"],
            Midpoint=prefill["midpoint"],
            Q_bias=params["q_bias"],
            K_bias=params["k_bias"],
            Angles=prefill["angles"],
            D=params["d"],
            Z=prefill["z"],
            Initial_States=None,
            chunk_size=args.chunk_size,
            return_final_states=True,
        )
        states = out[-1]
        for t in range(args.gen_len):
            _, states = simamba_siso_step(
                Q=decode["q"][:, t],
                K=decode["k"][:, t],
                V=decode["v"][:, t],
                ADT=decode["adt"][:, :, t],
                DT=decode["dt"][:, :, t],
                Simpson=decode["coeff"][:, :, t],
                Midpoint=(decode["midpoint"][:, :, t] if decode["midpoint"] is not None else None),
                Q_bias=params["q_bias"],
                K_bias=params["k_bias"],
                Angles=decode["angles"][:, t],
                D=params["d"],
                Z=decode["z"][:, t] if decode["z"] is not None else None,
                Input_States=states,
            )

    m3_e2e_once()
    sim_e2e_once()

    m3_ms = _time_ms(m3_e2e_once, repeats=args.e2e_repeats)
    sim_ms = _time_ms(sim_e2e_once, repeats=args.e2e_repeats)

    total_tokens = args.batch * (args.prompt_len + args.gen_len)
    gen_tokens = args.batch * args.gen_len

    m3_total_toks = total_tokens / (m3_ms * 1e-3)
    sim_total_toks = total_tokens / (sim_ms * 1e-3)
    m3_gen_toks = gen_tokens / (m3_ms * 1e-3)
    sim_gen_toks = gen_tokens / (sim_ms * 1e-3)

    print("\n[E2E] prompt+decode wall-clock")
    print(f"{'metric':<20} {'mamba3':>12} {'simamba':>12} {'sim/m3':>9} unit")
    _print_row("e2e_ms", m3_ms, sim_ms, "ms")
    _print_row("e2e_total_toks_s", m3_total_toks, sim_total_toks, "tok/s")
    _print_row("e2e_gen_toks_s", m3_gen_toks, sim_gen_toks, "tok/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Simamba (Simpson) vs Mamba-3 (trapezoidal) SISO kernels")
    parser.add_argument("--mode", choices=["micro", "e2e", "all"], default="all")

    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--gen-len", type=int, default=256)

    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--nheads-qk", type=int, default=None)
    parser.add_argument("--headdim-qk", type=int, default=64)
    parser.add_argument("--headdim-v", type=int, default=64)
    parser.add_argument("--n-angles", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-init-retries", type=int, default=3)
    parser.add_argument("--cuda-init-sleep", type=float, default=2.0)

    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--step-rep", type=int, default=100)
    parser.add_argument("--e2e-repeats", type=int, default=3)

    parser.add_argument("--no-z", action="store_true", help="Disable Z gating")
    parser.add_argument("--use-midpoint", action="store_true", help="Enable midpoint modulation for Simamba")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    if args.prompt_len <= 0:
        raise SystemExit("--prompt-len must be > 0")
    if args.gen_len <= 0:
        raise SystemExit("--gen-len must be > 0")
    if args.nheads_qk is None:
        args.nheads_qk = args.nheads
    if args.nheads % args.nheads_qk != 0:
        raise SystemExit("--nheads must be divisible by --nheads-qk")
    if args.headdim_qk % 2 != 0:
        raise SystemExit("--headdim-qk must be even")

    n_angles = args.n_angles if args.n_angles is not None else args.headdim_qk // 2
    if n_angles <= 0 or n_angles % 2 != 0:
        raise SystemExit("--n-angles must be a positive even number")
    if n_angles > args.headdim_qk // 2:
        raise SystemExit("--n-angles must be <= --headdim-qk // 2")

    dtype = _dtype_from_name(args.dtype)
    device = "cuda"

    torch.manual_seed(args.seed)
    _cuda_preflight(device=device, retries=args.cuda_init_retries, sleep_s=args.cuda_init_sleep)
    torch.cuda.manual_seed_all(args.seed)

    params = _retry_cuda_busy(
        op_name="parameter initialization",
        fn=lambda: {
            "q_bias": torch.randn(args.nheads, args.headdim_qk, device=device, dtype=torch.float32),
            "k_bias": torch.randn(args.nheads, args.headdim_qk, device=device, dtype=torch.float32),
            "d": torch.randn(args.nheads, device=device, dtype=torch.float32),
        },
        retries=args.cuda_init_retries,
        sleep_s=args.cuda_init_sleep,
    )

    prefill = _retry_cuda_busy(
        op_name="prefill tensor initialization",
        fn=lambda: _make_sequence_tensors(
            batch=args.batch,
            seqlen=args.prompt_len,
            nheads=args.nheads,
            nheads_qk=args.nheads_qk,
            headdim_qk=args.headdim_qk,
            headdim_v=args.headdim_v,
            n_angles=n_angles,
            dtype=dtype,
            use_z=not args.no_z,
            use_midpoint=args.use_midpoint,
            device=device,
        ),
        retries=args.cuda_init_retries,
        sleep_s=args.cuda_init_sleep,
    )
    decode = _retry_cuda_busy(
        op_name="decode tensor initialization",
        fn=lambda: _make_sequence_tensors(
            batch=args.batch,
            seqlen=args.gen_len,
            nheads=args.nheads,
            nheads_qk=args.nheads_qk,
            headdim_qk=args.headdim_qk,
            headdim_v=args.headdim_v,
            n_angles=n_angles,
            dtype=dtype,
            use_z=not args.no_z,
            use_midpoint=args.use_midpoint,
            device=device,
        ),
        retries=args.cuda_init_retries,
        sleep_s=args.cuda_init_sleep,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"mode={args.mode} batch={args.batch} prompt_len={args.prompt_len} gen_len={args.gen_len} "
        f"nheads={args.nheads} nheads_qk={args.nheads_qk} hqk={args.headdim_qk} hv={args.headdim_v} "
        f"n_angles={n_angles} dtype={args.dtype} chunk_size={args.chunk_size}"
    )
    print(
        f"warmup={args.warmup} rep={args.rep} step_rep={args.step_rep} e2e_repeats={args.e2e_repeats} "
        f"use_midpoint={args.use_midpoint} use_z={not args.no_z} cudagraph_step={do_bench_cudagraph is not None} "
        f"cuda_init_retries={args.cuda_init_retries} cuda_init_sleep={args.cuda_init_sleep}"
    )

    with torch.inference_mode():
        if args.mode in ("micro", "all"):
            _run_micro(
                args=args,
                params=params,
                prefill=prefill,
                decode=decode,
                nheads_qk=args.nheads_qk,
                n_angles=n_angles,
                dtype=dtype,
                device=device,
            )
        if args.mode in ("e2e", "all"):
            _run_e2e(args=args, params=params, prefill=prefill, decode=decode)


if __name__ == "__main__":
    main()
