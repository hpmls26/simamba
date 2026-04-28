"""Focused correctness checks for the chunk-parallel Simamba prefill path."""

import sys
import types
from pathlib import Path

import pytest
import torch

if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

if "mamba_ssm" not in sys.modules:
    repo_root = Path(__file__).resolve().parents[3]
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [str(repo_root / "mamba_ssm")]
    sys.modules["mamba_ssm"] = pkg

from mamba_ssm.ops.triton.simamba.mamba3_siso_fwd import mamba3_siso_fwd
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step


def _zero_states(
    batch: int,
    nheads: int,
    n_angles: int,
    headdim_qk: int,
    headdim_v: int,
    device: str,
    q_dtype: torch.dtype,
    v_dtype: torch.dtype,
):
    return (
        torch.zeros(batch, nheads, n_angles, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.zeros(batch, nheads, headdim_qk, device=device, dtype=q_dtype),
        torch.zeros(batch, nheads, headdim_qk, device=device, dtype=q_dtype),
        torch.zeros(batch, nheads, headdim_v, device=device, dtype=v_dtype),
        torch.zeros(batch, nheads, headdim_v, device=device, dtype=v_dtype),
    )


def _step_chunk_rollout(
    *,
    q,
    k,
    v,
    adt,
    dt,
    simpson,
    q_bias,
    k_bias,
    angles,
    midpoint=None,
    d=None,
    z=None,
    chunk_size: int,
    init_states=None,
):
    batch, seqlen, _, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape
    states = init_states
    if states is None:
        states = _zero_states(
            batch=batch,
            nheads=nheads,
            n_angles=angles.shape[-1],
            headdim_qk=headdim_qk,
            headdim_v=headdim_v,
            device=q.device,
            q_dtype=q.dtype,
            v_dtype=v.dtype,
        )

    outs = []
    chunk_states = []
    for t in range(seqlen):
        out_t, states = mamba3_siso_step(
            Q=q[:, t],
            K=k[:, t],
            V=v[:, t],
            ADT=adt[:, :, t],
            DT=dt[:, :, t],
            Simpson=simpson[:, :, t],
            Midpoint=midpoint[:, :, t] if midpoint is not None else None,
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles[:, t],
            D=d,
            Z=z[:, t] if z is not None else None,
            Input_States=states,
        )
        outs.append(out_t.unsqueeze(1))
        if (t + 1) % chunk_size == 0 or t == seqlen - 1:
            chunk_states.append((t, tuple(s.clone() for s in states)))
    return torch.cat(outs, dim=1), chunk_states


def _fused_chunk_rollout(
    *,
    q,
    k,
    v,
    adt,
    dt,
    simpson,
    q_bias,
    k_bias,
    angles,
    midpoint=None,
    d=None,
    z=None,
    chunk_size: int,
    init_states=None,
):
    seqlen = q.shape[1]
    states = init_states
    outs = []
    chunk_states = []
    for start in range(0, seqlen, chunk_size):
        end = min(seqlen, start + chunk_size)
        out, *_, states = mamba3_siso_fwd(
            Q=q[:, start:end],
            K=k[:, start:end],
            V=v[:, start:end],
            ADT=adt[:, :, start:end],
            DT=dt[:, :, start:end],
            Simpson=simpson[:, :, start:end],
            Midpoint=midpoint[:, :, start:end] if midpoint is not None else None,
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles[:, start:end],
            D=d,
            Z=z[:, start:end] if z is not None else None,
            Initial_States=states,
            chunk_size=chunk_size,
            store_states_adt_outv=False,
            return_final_states=True,
        )
        outs.append(out)
        chunk_states.append((end - 1, tuple(s.clone() for s in states)))
    return torch.cat(outs, dim=1), chunk_states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "term,max_diff",
    [
        ("gamma0", 7e-4),
        ("gamma1", 5e-7),
        ("gamma2", 5e-7),
    ],
)
def test_simamba_chunk_parallel_term_isolation(term: str, max_diff: float):
    torch.manual_seed(1700)
    device = "cuda"

    batch = 1
    seqlen = 1
    nheads = 2
    headdim_qk = 64
    headdim_v = 64
    n_angles = headdim_qk // 2

    q = torch.zeros(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16)
    k = torch.zeros(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16)
    v = torch.zeros(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)
    adt = 0.05 * torch.randn(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.05 + 0.1 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = 0.25 + 0.5 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    q_bias = torch.zeros(nheads, headdim_qk, device=device, dtype=torch.bfloat16)
    k_bias = torch.zeros(nheads, headdim_qk, device=device, dtype=torch.bfloat16)
    angles = torch.zeros(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)

    init_states = list(
        _zero_states(
            batch=batch,
            nheads=nheads,
            n_angles=n_angles,
            headdim_qk=headdim_qk,
            headdim_v=headdim_v,
            device=device,
            q_dtype=q.dtype,
            v_dtype=v.dtype,
        )
    )

    if term == "gamma0":
        k.normal_()
        v.normal_()
    elif term == "gamma1":
        init_states[2].normal_()
        init_states[4].normal_()
    elif term == "gamma2":
        init_states[3].normal_()
        init_states[5].normal_()
    else:
        raise AssertionError(f"Unexpected term: {term}")

    init_states = tuple(init_states)
    _, *_, final_states = mamba3_siso_fwd(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        Initial_States=init_states,
        chunk_size=64,
        store_states_adt_outv=False,
        return_final_states=True,
    )
    assert final_states is not None

    alpha = torch.exp(adt[:, :, 0])
    alpha_half = torch.exp(adt[:, :, 0] * 0.5)
    sim = simpson[:, :, 0].clamp(0.0, 1.0)
    g0 = (dt[:, :, 0] / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * sim))
    g1 = (dt[:, :, 0] / 6.0) * (alpha + alpha_half * (2.0 + sim))
    g2 = -(dt[:, :, 0] / 12.0) * alpha_half * sim

    expected = alpha[:, :, None, None] * init_states[1]
    if term == "gamma0":
        expected = expected + g0[:, :, None, None] * (
            v[:, 0].float()[:, :, :, None] * k[:, 0].float()[:, :, None, :]
        )
    elif term == "gamma1":
        expected = expected + g1[:, :, None, None] * (
            init_states[4].float()[:, :, :, None] * init_states[2].float()[:, :, None, :]
        )
    else:
        expected = expected + g2[:, :, None, None] * (
            init_states[5].float()[:, :, :, None] * init_states[3].float()[:, :, None, :]
        )

    diff = (final_states[1] - expected).abs()
    assert diff.max().item() < max_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_chunk_parallel_boundary_lag_states_match_step_rollout():
    torch.manual_seed(1801)
    device = "cuda"

    batch = 1
    seqlen = 129
    chunk_size = 64
    nheads_qk = 2
    nheads = 4
    headdim_qk = 16
    headdim_v = 16
    n_angles = 8

    q = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)
    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    midpoint = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    init_states = (
        torch.randn(batch, nheads, n_angles, device=device, dtype=torch.float32),
        torch.randn(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16),
    )

    _, fused_states = _fused_chunk_rollout(
        q=q,
        k=k,
        v=v,
        adt=adt,
        dt=dt,
        simpson=simpson,
        q_bias=q_bias,
        k_bias=k_bias,
        angles=angles,
        midpoint=midpoint,
        chunk_size=chunk_size,
        init_states=init_states,
    )
    _, step_states = _step_chunk_rollout(
        q=q,
        k=k,
        v=v,
        adt=adt,
        dt=dt,
        simpson=simpson,
        q_bias=q_bias,
        k_bias=k_bias,
        angles=angles,
        midpoint=midpoint,
        chunk_size=chunk_size,
        init_states=init_states,
    )

    for (chunk_end_fused, fused), (chunk_end_step, step) in zip(fused_states, step_states):
        assert chunk_end_fused == chunk_end_step
        for fused_state, step_state in zip(fused[2:], step[2:]):
            diff = (fused_state.float() - step_state.float()).abs()
            assert diff.max().item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_chunk_parallel_chunk_end_states_close_to_step_rollout():
    torch.manual_seed(1902)
    device = "cuda"

    batch = 1
    seqlen = 129
    chunk_size = 64
    nheads_qk = 2
    nheads = 4
    headdim_qk = 16
    headdim_v = 16
    n_angles = 8

    q = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)
    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    midpoint = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    d = torch.randn(nheads, device=device, dtype=torch.float32)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)
    init_states = (
        torch.randn(batch, nheads, n_angles, device=device, dtype=torch.float32),
        torch.randn(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16),
    )

    out_fused, fused_states = _fused_chunk_rollout(
        q=q,
        k=k,
        v=v,
        adt=adt,
        dt=dt,
        simpson=simpson,
        q_bias=q_bias,
        k_bias=k_bias,
        angles=angles,
        midpoint=midpoint,
        d=d,
        z=z,
        chunk_size=chunk_size,
        init_states=init_states,
    )
    out_step, step_states = _step_chunk_rollout(
        q=q,
        k=k,
        v=v,
        adt=adt,
        dt=dt,
        simpson=simpson,
        q_bias=q_bias,
        k_bias=k_bias,
        angles=angles,
        midpoint=midpoint,
        d=d,
        z=z,
        chunk_size=chunk_size,
        init_states=init_states,
    )

    out_diff = (out_fused.float() - out_step.float()).abs()
    assert out_diff.max().item() < 7e-1

    for (chunk_end_fused, fused), (chunk_end_step, step) in zip(fused_states, step_states):
        assert chunk_end_fused == chunk_end_step
        angle_diff = (fused[0].float() - step[0].float()).abs()
        ssm_diff = (fused[1].float() - step[1].float()).abs()
        assert angle_diff.max().item() < 3e-2
        assert ssm_diff.max().item() < 1e-1
