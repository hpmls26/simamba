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
