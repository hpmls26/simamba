"""Simamba Triton coefficient-backward tests."""

import sys
import types
from pathlib import Path

import pytest
import torch

# Avoid importing mamba_ssm/__init__.py for focused kernel tests.
if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

if "mamba_ssm" not in sys.modules:
    repo_root = Path(__file__).resolve().parents[3]
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [str(repo_root / "mamba_ssm")]
    sys.modules["mamba_ssm"] = pkg

from mamba_ssm.ops.triton.simamba.mamba3_siso_bwd import compute_dcoeffs
from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (
    _reference_autograd_grads,
    mamba3_siso_combined as simamba_triton_siso_combined,
)
from mamba_ssm.modules.simamba import Simamba
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import simamba_siso_combined


def _finite_diff(loss_fn, tensor, idx, eps=1e-3):
    plus = tensor.clone()
    minus = tensor.clone()
    plus[idx] += eps
    minus[idx] -= eps
    fp = loss_fn(plus)
    fm = loss_fn(minus)
    return (fp - fm) / (2.0 * eps)


def _assert_close_tensor(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float):
    max_abs = (got.float() - ref.float()).abs().max().item()
    assert torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol), (
        f"{name} mismatch: max_abs={max_abs:.4e}, atol={atol}, rtol={rtol}"
    )


def test_simamba_reference_autograd_chunked_matches_direct_autograd():
    torch.manual_seed(21)

    batch, seqlen = 1, 5
    nheads = 2
    headdim_qk = 4
    headdim_v = 3
    n_angles = 2

    q = torch.randn(batch, seqlen, nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, requires_grad=True)
    adt = torch.randn(batch, nheads, seqlen, dtype=torch.float32)
    dt = (0.1 + torch.rand(batch, nheads, seqlen, dtype=torch.float32)).requires_grad_(False)
    simpson = torch.rand(batch, nheads, seqlen, dtype=torch.float32)
    q_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    k_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(batch, seqlen, nheads, n_angles, dtype=torch.float32, requires_grad=True)
    d = torch.randn(nheads, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32)

    helper_grads = _reference_autograd_grads(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=None,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        Input_States=None,
        grad_out=grad_out,
        grad_final_angle_state=None,
        grad_final_ssm_state=None,
        grad_final_k_prev1_state=None,
        grad_final_k_prev2_state=None,
        grad_final_v_prev1_state=None,
        grad_final_v_prev2_state=None,
        return_final_states=False,
        needs_grad={
            "Q": True,
            "K": True,
            "V": True,
            "ADT": False,
            "DT": False,
            "Simpson": False,
            "Midpoint": False,
            "Q_bias": True,
            "K_bias": True,
            "Angles": True,
            "D": True,
            "Z": True,
        },
        recompute_chunk_size=2,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_bias_ref = q_bias.detach().clone().requires_grad_(True)
    k_bias_ref = k_bias.detach().clone().requires_grad_(True)
    angles_ref = angles.detach().clone().requires_grad_(True)
    d_ref = d.detach().clone().requires_grad_(True)
    z_ref = z.detach().clone().requires_grad_(True)

    out_ref = simamba_siso_combined(
        Q=q_ref,
        K=k_ref,
        V=v_ref,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=None,
        Q_bias=q_bias_ref,
        K_bias=k_bias_ref,
        Angles=angles_ref,
        D=d_ref,
        Z=z_ref,
    )
    torch.autograd.backward(out_ref, grad_tensors=grad_out)

    checks = {
        "Q": q_ref.grad,
        "K": k_ref.grad,
        "V": v_ref.grad,
        "Q_bias": q_bias_ref.grad,
        "K_bias": k_bias_ref.grad,
        "Angles": angles_ref.grad,
        "D": d_ref.grad,
        "Z": z_ref.grad,
    }
    for name, ref in checks.items():
        _assert_close_tensor(f"chunked_ref.{name}", helper_grads[name], ref, atol=1e-5, rtol=1e-4)


def test_simamba_reference_autograd_chunked_matches_direct_autograd_with_states():
    torch.manual_seed(22)

    batch, seqlen = 1, 4
    nheads = 2
    headdim_qk = 4
    headdim_v = 3
    n_angles = 2

    q = torch.randn(batch, seqlen, nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, requires_grad=True)
    adt = torch.randn(batch, nheads, seqlen, dtype=torch.float32, requires_grad=True)
    dt = (0.1 + torch.rand(batch, nheads, seqlen, dtype=torch.float32)).requires_grad_(True)
    simpson = torch.rand(batch, nheads, seqlen, dtype=torch.float32, requires_grad=True)
    midpoint = torch.rand(batch, nheads, seqlen, dtype=torch.float32, requires_grad=True)
    q_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    k_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(batch, seqlen, nheads, n_angles, dtype=torch.float32, requires_grad=True)
    d = torch.randn(nheads, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, requires_grad=True)
    input_states = (
        torch.randn(batch, nheads, n_angles, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, headdim_qk, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_qk, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_qk, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, dtype=torch.float32, requires_grad=True),
    )
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32)

    out_ref = simamba_siso_combined(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=midpoint,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        Input_States=input_states,
        return_final_states=True,
    )
    grad_finals = tuple(torch.randn_like(tensor) for tensor in out_ref[1:])

    helper_grads = _reference_autograd_grads(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=midpoint,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        Input_States=input_states,
        grad_out=grad_out,
        grad_final_angle_state=grad_finals[0],
        grad_final_ssm_state=grad_finals[1],
        grad_final_k_prev1_state=grad_finals[2],
        grad_final_k_prev2_state=grad_finals[3],
        grad_final_v_prev1_state=grad_finals[4],
        grad_final_v_prev2_state=grad_finals[5],
        return_final_states=True,
        needs_grad={
            "Q": True,
            "K": True,
            "V": True,
            "ADT": True,
            "DT": True,
            "Simpson": True,
            "Midpoint": True,
            "Q_bias": True,
            "K_bias": True,
            "Angles": True,
            "D": True,
            "Z": True,
            "Input_Angle_State": True,
            "Input_SSM_State": True,
            "Input_K_Prev1_State": True,
            "Input_K_Prev2_State": True,
            "Input_V_Prev1_State": True,
            "Input_V_Prev2_State": True,
        },
        recompute_chunk_size=2,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    adt_ref = adt.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)
    simpson_ref = simpson.detach().clone().requires_grad_(True)
    midpoint_ref = midpoint.detach().clone().requires_grad_(True)
    q_bias_ref = q_bias.detach().clone().requires_grad_(True)
    k_bias_ref = k_bias.detach().clone().requires_grad_(True)
    angles_ref = angles.detach().clone().requires_grad_(True)
    d_ref = d.detach().clone().requires_grad_(True)
    z_ref = z.detach().clone().requires_grad_(True)
    input_states_ref = tuple(state.detach().clone().requires_grad_(True) for state in input_states)

    out_direct = simamba_siso_combined(
        Q=q_ref,
        K=k_ref,
        V=v_ref,
        ADT=adt_ref,
        DT=dt_ref,
        Simpson=simpson_ref,
        Midpoint=midpoint_ref,
        Q_bias=q_bias_ref,
        K_bias=k_bias_ref,
        Angles=angles_ref,
        D=d_ref,
        Z=z_ref,
        Input_States=input_states_ref,
        return_final_states=True,
    )
    torch.autograd.backward(out_direct, grad_tensors=(grad_out, *grad_finals))

    checks = {
        "Q": q_ref.grad,
        "K": k_ref.grad,
        "V": v_ref.grad,
        "ADT": adt_ref.grad,
        "DT": dt_ref.grad,
        "Simpson": simpson_ref.grad,
        "Midpoint": midpoint_ref.grad,
        "Q_bias": q_bias_ref.grad,
        "K_bias": k_bias_ref.grad,
        "Angles": angles_ref.grad,
        "D": d_ref.grad,
        "Z": z_ref.grad,
        "Input_Angle_State": input_states_ref[0].grad,
        "Input_SSM_State": input_states_ref[1].grad,
        "Input_K_Prev1_State": input_states_ref[2].grad,
        "Input_K_Prev2_State": input_states_ref[3].grad,
        "Input_V_Prev1_State": input_states_ref[4].grad,
        "Input_V_Prev2_State": input_states_ref[5].grad,
    }
    for name, ref in checks.items():
        _assert_close_tensor(f"chunked_ref_state.{name}", helper_grads[name], ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_dcoeffs_match_finite_difference_with_midpoint():
    torch.manual_seed(0)
    device = "cuda"

    batch, seqlen = 1, 5
    nheads = 2
    headdim_qk = 4
    headdim_v = 4
    n_angles = 2

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)

    adt = (-0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).clamp(-1.0, -1e-3)
    dt = (0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).clamp(1e-3, 1.0)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    midpoint = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)

    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)

    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)

    dadt, ddt, dsimpson, dmidpoint = compute_dcoeffs(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=midpoint,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=None,
        Z=z,
        grad_out=grad_out,
    )

    assert dmidpoint is not None

    def _loss(adt_, dt_, simpson_, midpoint_):
        out = simamba_siso_combined(
            Q=q,
            K=k,
            V=v,
            ADT=adt_,
            DT=dt_,
            Simpson=simpson_,
            Midpoint=midpoint_,
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles,
            D=None,
            Z=z,
        )
        return (out.float() * grad_out).sum().item()

    sample_indices = [(0, 0, 0), (0, 1, 2), (0, 0, 4)]

    for idx in sample_indices:
        num = _finite_diff(lambda x: _loss(x, dt, simpson, midpoint), adt, idx)
        ana = float(dadt[idx].item())
        assert abs(ana - num) < 6e-2

        num = _finite_diff(lambda x: _loss(adt, x, simpson, midpoint), dt, idx)
        ana = float(ddt[idx].item())
        assert abs(ana - num) < 6e-2

        num = _finite_diff(lambda x: _loss(adt, dt, x, midpoint), simpson, idx)
        ana = float(dsimpson[idx].item())
        assert abs(ana - num) < 6e-2

        num = _finite_diff(lambda x: _loss(adt, dt, simpson, x), midpoint, idx)
        ana = float(dmidpoint[idx].item())
        assert abs(ana - num) < 6e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_dcoeffs_without_midpoint_returns_none_for_dmidpoint():
    torch.manual_seed(1)
    device = "cuda"

    batch, seqlen = 1, 4
    nheads = 2
    headdim_qk = 4
    headdim_v = 4
    n_angles = 2

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)

    dadt, ddt, dsimpson, dmidpoint = compute_dcoeffs(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=None,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=None,
        Z=None,
        grad_out=grad_out,
    )

    assert dadt.shape == adt.shape
    assert ddt.shape == dt.shape
    assert dsimpson.shape == simpson.shape
    assert dmidpoint is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_dcoeffs_match_reference_autograd_full_tensor_with_midpoint():
    torch.manual_seed(9)
    device = "cuda"

    batch, seqlen = 2, 6
    nheads = 3
    headdim_qk = 8
    headdim_v = 6
    n_angles = 4

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    midpoint = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)

    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    d = torch.randn(nheads, device=device, dtype=torch.float32)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)

    dadt, ddt, dsimpson, dmidpoint = compute_dcoeffs(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=midpoint,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        grad_out=grad_out,
    )
    assert dmidpoint is not None

    adt_ref = adt.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)
    simpson_ref = simpson.detach().clone().requires_grad_(True)
    midpoint_ref = midpoint.detach().clone().requires_grad_(True)

    out_ref = simamba_siso_combined(
        Q=q,
        K=k,
        V=v,
        ADT=adt_ref,
        DT=dt_ref,
        Simpson=simpson_ref,
        Midpoint=midpoint_ref,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        return_final_states=False,
    )
    torch.autograd.backward(out_ref, grad_tensors=grad_out)

    _assert_close_tensor("dADT", dadt, adt_ref.grad, atol=1e-1, rtol=2.5e-1)
    _assert_close_tensor("dDT", ddt, dt_ref.grad, atol=1.2e-1, rtol=3.0e-1)
    _assert_close_tensor("dSimpson", dsimpson, simpson_ref.grad, atol=1e-1, rtol=2.5e-1)
    _assert_close_tensor("dMidpoint", dmidpoint, midpoint_ref.grad, atol=1e-1, rtol=2.5e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_dcoeffs_match_reference_autograd_full_tensor_without_midpoint():
    torch.manual_seed(10)
    device = "cuda"

    batch, seqlen = 2, 6
    nheads = 2
    headdim_qk = 8
    headdim_v = 6
    n_angles = 4

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.float32)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    adt = -0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)

    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    d = torch.randn(nheads, device=device, dtype=torch.float32)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)
    grad_out = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.float32)

    dadt, ddt, dsimpson, dmidpoint = compute_dcoeffs(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=None,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        grad_out=grad_out,
    )
    assert dmidpoint is None

    adt_ref = adt.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)
    simpson_ref = simpson.detach().clone().requires_grad_(True)

    out_ref = simamba_siso_combined(
        Q=q,
        K=k,
        V=v,
        ADT=adt_ref,
        DT=dt_ref,
        Simpson=simpson_ref,
        Midpoint=None,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        return_final_states=False,
    )
    torch.autograd.backward(out_ref, grad_tensors=grad_out)

    _assert_close_tensor("dADT(no_mid)", dadt, adt_ref.grad, atol=1e-1, rtol=2.5e-1)
    _assert_close_tensor("dDT(no_mid)", ddt, dt_ref.grad, atol=1.2e-1, rtol=3.0e-1)
    _assert_close_tensor("dSimpson(no_mid)", dsimpson, simpson_ref.grad, atol=1e-1, rtol=2.5e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_triton_combined_matches_reference_autograd():
    torch.manual_seed(12)
    device = "cuda"

    batch, seqlen = 2, 16
    nheads = 2
    headdim_qk = 16
    headdim_v = 8
    n_angles = 4

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
    adt = (-0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).requires_grad_(True)
    dt = (0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).requires_grad_(True)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.bfloat16, requires_grad=True)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32, requires_grad=True)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.bfloat16, requires_grad=True)
    d = torch.randn(nheads, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)

    out = simamba_triton_siso_combined(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        chunk_size=16,
    )
    loss = out.float().square().mean()
    loss.backward()
    grads_triton = {
        "Q": q.grad.detach().clone(),
        "K": k.grad.detach().clone(),
        "V": v.grad.detach().clone(),
        "ADT": adt.grad.detach().clone(),
        "DT": dt.grad.detach().clone(),
        "Simpson": simpson.grad.detach().clone(),
        "Q_bias": q_bias.grad.detach().clone(),
        "K_bias": k_bias.grad.detach().clone(),
        "Angles": angles.grad.detach().clone(),
        "D": d.grad.detach().clone(),
        "Z": z.grad.detach().clone(),
    }

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    adt_ref = adt.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)
    simpson_ref = simpson.detach().clone().requires_grad_(True)
    q_bias_ref = q_bias.detach().clone().requires_grad_(True)
    k_bias_ref = k_bias.detach().clone().requires_grad_(True)
    angles_ref = angles.detach().clone().requires_grad_(True)
    d_ref = d.detach().clone().requires_grad_(True)
    z_ref = z.detach().clone().requires_grad_(True)

    out_ref = simamba_siso_combined(
        Q=q_ref,
        K=k_ref,
        V=v_ref,
        ADT=adt_ref,
        DT=dt_ref,
        Simpson=simpson_ref,
        Midpoint=None,
        Q_bias=q_bias_ref,
        K_bias=k_bias_ref,
        Angles=angles_ref,
        D=d_ref,
        Z=z_ref,
    )
    loss_ref = out_ref.float().square().mean()
    loss_ref.backward()
    grads_ref = {
        "Q": q_ref.grad,
        "K": k_ref.grad,
        "V": v_ref.grad,
        "ADT": adt_ref.grad,
        "DT": dt_ref.grad,
        "Simpson": simpson_ref.grad,
        "Q_bias": q_bias_ref.grad,
        "K_bias": k_bias_ref.grad,
        "Angles": angles_ref.grad,
        "D": d_ref.grad,
        "Z": z_ref.grad,
    }

    for name in grads_ref:
        _assert_close_tensor(name, grads_triton[name], grads_ref[name], atol=2e-1, rtol=3.5e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_module_triton_backward_uses_core_parameters():
    torch.manual_seed(13)
    device = "cuda"

    model = Simamba(
        d_model=64,
        d_state=16,
        expand=2,
        headdim=16,
        ngroups=1,
        rope_fraction=0.5,
        chunk_size=16,
        simamba_backend="triton",
        device=device,
        dtype=torch.bfloat16,
    )
    model.train()

    u = torch.randn(2, 32, 64, device=device, dtype=torch.bfloat16)
    loss = model(u).float().square().mean()
    loss.backward()

    assert model.in_proj.weight.grad is not None
    assert model.dt_bias.grad is not None
    assert model.B_bias.grad is not None
    assert model.C_bias.grad is not None
    assert model.B_norm.weight.grad is not None
    assert model.C_norm.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_triton_combined_matches_reference_autograd_with_midpoint_and_states():
    torch.manual_seed(14)
    device = "cuda"

    batch, seqlen = 1, 16
    nheads = 2
    headdim_qk = 16
    headdim_v = 8
    n_angles = 4

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
    adt = (-0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).requires_grad_(True)
    dt = (0.01 + 0.2 * torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32)).requires_grad_(True)
    simpson = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.bfloat16, requires_grad=True)
    midpoint = torch.rand(batch, nheads, seqlen, device=device, dtype=torch.bfloat16, requires_grad=True)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32, requires_grad=True)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(batch, seqlen, nheads, n_angles, device=device, dtype=torch.bfloat16, requires_grad=True)
    d = torch.randn(nheads, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_states = (
        torch.randn(batch, nheads, n_angles, device=device, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32, requires_grad=True),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True),
        torch.randn(batch, nheads, headdim_qk, device=device, dtype=torch.bfloat16, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True),
        torch.randn(batch, nheads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True),
    )

    outputs = simamba_triton_siso_combined(
        Q=q,
        K=k,
        V=v,
        ADT=adt,
        DT=dt,
        Simpson=simpson,
        Midpoint=midpoint,
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles,
        D=d,
        Z=z,
        Initial_States=input_states,
        return_final_states=True,
        chunk_size=16,
    )
    grad_out = torch.randn_like(outputs[0])
    grad_finals = tuple(torch.randn_like(tensor) for tensor in outputs[1:])
    torch.autograd.backward(outputs, grad_tensors=(grad_out, *grad_finals))
    grads_triton = {
        "Q": q.grad.detach().clone(),
        "K": k.grad.detach().clone(),
        "V": v.grad.detach().clone(),
        "ADT": adt.grad.detach().clone(),
        "DT": dt.grad.detach().clone(),
        "Simpson": simpson.grad.detach().clone(),
        "Midpoint": midpoint.grad.detach().clone(),
        "Q_bias": q_bias.grad.detach().clone(),
        "K_bias": k_bias.grad.detach().clone(),
        "Angles": angles.grad.detach().clone(),
        "D": d.grad.detach().clone(),
        "Z": z.grad.detach().clone(),
        "Input_Angle_State": input_states[0].grad.detach().clone(),
        "Input_SSM_State": input_states[1].grad.detach().clone(),
        "Input_K_Prev1_State": input_states[2].grad.detach().clone(),
        "Input_K_Prev2_State": input_states[3].grad.detach().clone(),
        "Input_V_Prev1_State": input_states[4].grad.detach().clone(),
        "Input_V_Prev2_State": input_states[5].grad.detach().clone(),
    }

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    adt_ref = adt.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)
    simpson_ref = simpson.detach().clone().requires_grad_(True)
    midpoint_ref = midpoint.detach().clone().requires_grad_(True)
    q_bias_ref = q_bias.detach().clone().requires_grad_(True)
    k_bias_ref = k_bias.detach().clone().requires_grad_(True)
    angles_ref = angles.detach().clone().requires_grad_(True)
    d_ref = d.detach().clone().requires_grad_(True)
    z_ref = z.detach().clone().requires_grad_(True)
    input_states_ref = tuple(state.detach().clone().requires_grad_(True) for state in input_states)

    outputs_ref = simamba_siso_combined(
        Q=q_ref,
        K=k_ref,
        V=v_ref,
        ADT=adt_ref,
        DT=dt_ref,
        Simpson=simpson_ref,
        Midpoint=midpoint_ref,
        Q_bias=q_bias_ref,
        K_bias=k_bias_ref,
        Angles=angles_ref,
        D=d_ref,
        Z=z_ref,
        Input_States=input_states_ref,
        return_final_states=True,
    )
    torch.autograd.backward(outputs_ref, grad_tensors=(grad_out, *grad_finals))
    grads_ref = {
        "Q": q_ref.grad,
        "K": k_ref.grad,
        "V": v_ref.grad,
        "ADT": adt_ref.grad,
        "DT": dt_ref.grad,
        "Simpson": simpson_ref.grad,
        "Midpoint": midpoint_ref.grad,
        "Q_bias": q_bias_ref.grad,
        "K_bias": k_bias_ref.grad,
        "Angles": angles_ref.grad,
        "D": d_ref.grad,
        "Z": z_ref.grad,
        "Input_Angle_State": input_states_ref[0].grad,
        "Input_SSM_State": input_states_ref[1].grad,
        "Input_K_Prev1_State": input_states_ref[2].grad,
        "Input_K_Prev2_State": input_states_ref[3].grad,
        "Input_V_Prev1_State": input_states_ref[4].grad,
        "Input_V_Prev2_State": input_states_ref[5].grad,
    }

    for name in grads_ref:
        _assert_close_tensor(name, grads_triton[name], grads_ref[name], atol=2.5e-1, rtol=4.0e-1)
