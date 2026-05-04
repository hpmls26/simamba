"""Simamba Triton SISO forward parity tests."""

import sys
import types
from pathlib import Path
from argparse import Namespace

import pytest
import torch

# Avoid importing mamba_ssm/__init__.py during these focused kernel tests,
# which otherwise requires the compiled selective_scan_cuda extension.
if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

if "mamba_ssm" not in sys.modules:
    repo_root = Path(__file__).resolve().parents[3]
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [str(repo_root / "mamba_ssm")]
    sys.modules["mamba_ssm"] = pkg

from mamba_ssm.ops.triton.simamba.mamba3_siso_fwd import mamba3_siso_fwd
from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (
    mamba3_siso_combined as simamba_triton_siso_combined,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step
from mamba_ssm.modules.simamba import Simamba
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import simamba_siso_combined
from scripts.train_simamba_lm import simamba_correction_scale_for_step


def test_simamba_correction_scale_schedule():
    args = Namespace(
        model_layer="Simamba",
        simamba_discretization="simpson",
        simamba_correction_anneal_min=0.0,
        simamba_correction_anneal_max=1.0,
        simamba_correction_anneal_start=10,
        simamba_correction_anneal_steps=100,
        simamba_correction_anneal_schedule="linear",
    )

    assert simamba_correction_scale_for_step(0, args) == 0.0
    assert simamba_correction_scale_for_step(10, args) == 0.0
    assert simamba_correction_scale_for_step(60, args) == 0.5
    assert simamba_correction_scale_for_step(110, args) == 1.0
    assert simamba_correction_scale_for_step(1000, args) == 1.0

    args.simamba_correction_anneal_schedule = "cosine"
    assert simamba_correction_scale_for_step(60, args) == pytest.approx(0.5)

    args.model_layer = "Mamba2"
    assert simamba_correction_scale_for_step(60, args) == 1.0


def test_simamba_module_scales_and_clamps_simpson_correction():
    model = Simamba(
        d_model=64,
        d_state=16,
        expand=2,
        headdim=16,
        simamba_backend="reference",
        device="cpu",
        dtype=torch.float32,
    )
    simpson = torch.ones(2, 3, dtype=torch.float32)

    assert torch.allclose(model._scale_simpson_correction(simpson), simpson)

    model.set_simpson_correction_scale(0.25)
    assert model.get_simpson_correction_scale() == pytest.approx(0.25)
    assert torch.allclose(model._scale_simpson_correction(simpson), simpson * 0.25)

    model.set_simpson_correction_scale(-4.0)
    assert model.get_simpson_correction_scale() == 0.0
    assert torch.count_nonzero(model._scale_simpson_correction(simpson)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_triton_forward_matches_reference_with_midpoint():
    torch.manual_seed(123)
    device = "cuda"

    batch, seqlen = 2, 8
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4

    q = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, headdim_qk, device=device, dtype=torch.bfloat16)
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

    ref = simamba_siso_combined(
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
        return_final_states=True,
    )

    tri = mamba3_siso_fwd(
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
        return_final_states=True,
    )

    out_ref = ref[0].float()
    out_tri = tri[0].float()
    assert (out_ref - out_tri).abs().max().item() < 5e-2

    final_ref = ref[1:]
    final_tri = tri[-1]
    assert final_tri is not None

    assert (final_ref[0].float() - final_tri[0].float()).abs().max().item() < 5e-3
    assert (final_ref[1].float() - final_tri[1].float()).abs().max().item() < 8e-2
    assert (final_ref[2].float() - final_tri[2].float()).abs().max().item() < 5e-2
    assert (final_ref[3].float() - final_tri[3].float()).abs().max().item() < 5e-2
    assert (final_ref[4].float() - final_tri[4].float()).abs().max().item() < 5e-2
    assert (final_ref[5].float() - final_tri[5].float()).abs().max().item() < 5e-2


def test_simamba_triton_forward_varlen_is_rejected():
    torch.manual_seed(321)
    device = "cpu"

    batch = 1
    nheads_qk = 2
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4
    total_seqlen = 10
    cu_seqlens = torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int32)

    q = torch.randn(batch, total_seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, total_seqlen, nheads_qk, headdim_qk, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, total_seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)
    adt = -0.2 * torch.rand(batch, nheads, total_seqlen, device=device, dtype=torch.float32)
    dt = 0.01 + 0.2 * torch.rand(batch, nheads, total_seqlen, device=device, dtype=torch.float32)
    simpson = torch.rand(batch, nheads, total_seqlen, device=device, dtype=torch.float32)
    midpoint = torch.rand(batch, nheads, total_seqlen, device=device, dtype=torch.float32)
    q_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    k_bias = torch.randn(nheads, headdim_qk, device=device, dtype=torch.float32)
    angles = torch.randn(batch, total_seqlen, nheads, n_angles, device=device, dtype=torch.float32)
    d = torch.randn(nheads, device=device, dtype=torch.float32)
    z = torch.randn(batch, total_seqlen, nheads, headdim_v, device=device, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        mamba3_siso_fwd(
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
            return_final_states=True,
            cu_seqlens=cu_seqlens,
        )

    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        simamba_triton_siso_combined(
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
            return_final_states=True,
            cu_seqlens=cu_seqlens,
        )


def test_simamba_module_varlen_is_rejected():
    model = Simamba(
        d_model=64,
        d_state=16,
        expand=2,
        headdim=16,
        simamba_backend="reference",
        device="cpu",
        dtype=torch.float32,
    )
    u = torch.randn(1, 8, 64, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        model(u, cu_seqlens=cu_seqlens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_triton_prefill_step_decode_parity():
    torch.manual_seed(777)
    device = "cuda"

    batch = 2
    seqlen = 9
    prefill_len = 4
    nheads_qk = 2
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4

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

    full = mamba3_siso_fwd(
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
        return_final_states=False,
    )
    out_full = full[0]

    prefill = mamba3_siso_fwd(
        Q=q[:, :prefill_len],
        K=k[:, :prefill_len],
        V=v[:, :prefill_len],
        ADT=adt[:, :, :prefill_len],
        DT=dt[:, :, :prefill_len],
        Simpson=simpson[:, :, :prefill_len],
        Midpoint=midpoint[:, :, :prefill_len],
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles[:, :prefill_len],
        D=d,
        Z=z[:, :prefill_len],
        return_final_states=True,
    )
    out_prefill = prefill[0]
    states = prefill[-1]
    assert states is not None

    decode_chunks = []
    for t in range(prefill_len, seqlen):
        out_t, states = mamba3_siso_step(
            Q=q[:, t],
            K=k[:, t],
            V=v[:, t],
            ADT=adt[:, :, t],
            DT=dt[:, :, t],
            Simpson=simpson[:, :, t],
            Midpoint=midpoint[:, :, t],
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles[:, t],
            D=d,
            Z=z[:, t],
            Input_States=states,
        )
        decode_chunks.append(out_t.unsqueeze(1))

    out_decode = torch.cat(decode_chunks, dim=1)
    out_rollout = torch.cat([out_prefill, out_decode], dim=1)

    assert (out_full.float() - out_rollout.float()).abs().max().item() < 6e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seqlen", [1, 2])
def test_simamba_triton_forward_short_sequence_edges(seqlen: int):
    torch.manual_seed(901 + seqlen)
    device = "cuda"

    batch = 2
    nheads_qk = 2
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4

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

    ref = simamba_siso_combined(
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
        return_final_states=True,
    )

    tri = mamba3_siso_fwd(
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
        return_final_states=True,
    )

    out_ref = ref[0].float()
    out_tri = tri[0].float()
    assert (out_ref - out_tri).abs().max().item() < 7e-2

    final_ref = ref[1:]
    final_tri = tri[-1]
    assert final_tri is not None

    assert (final_ref[0].float() - final_tri[0].float()).abs().max().item() < 1e-2
    assert (final_ref[1].float() - final_tri[1].float()).abs().max().item() < 9e-2
    assert (final_ref[2].float() - final_tri[2].float()).abs().max().item() < 7e-2
    assert (final_ref[3].float() - final_tri[3].float()).abs().max().item() < 7e-2
    assert (final_ref[4].float() - final_tri[4].float()).abs().max().item() < 7e-2
    assert (final_ref[5].float() - final_tri[5].float()).abs().max().item() < 7e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seqlen,prefill_len", [(1, 1), (2, 1)])
def test_simamba_triton_prefill_step_decode_parity_short_sequences(seqlen: int, prefill_len: int):
    torch.manual_seed(1100 + seqlen)
    device = "cuda"

    batch = 2
    nheads_qk = 2
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4

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

    full = mamba3_siso_fwd(
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
        return_final_states=True,
    )
    out_full = full[0]
    full_states = full[-1]
    assert full_states is not None

    prefill = mamba3_siso_fwd(
        Q=q[:, :prefill_len],
        K=k[:, :prefill_len],
        V=v[:, :prefill_len],
        ADT=adt[:, :, :prefill_len],
        DT=dt[:, :, :prefill_len],
        Simpson=simpson[:, :, :prefill_len],
        Midpoint=midpoint[:, :, :prefill_len],
        Q_bias=q_bias,
        K_bias=k_bias,
        Angles=angles[:, :prefill_len],
        D=d,
        Z=z[:, :prefill_len],
        return_final_states=True,
    )
    out_prefill = prefill[0]
    states = prefill[-1]
    assert states is not None

    decode_chunks = []
    for t in range(prefill_len, seqlen):
        out_t, states = mamba3_siso_step(
            Q=q[:, t],
            K=k[:, t],
            V=v[:, t],
            ADT=adt[:, :, t],
            DT=dt[:, :, t],
            Simpson=simpson[:, :, t],
            Midpoint=midpoint[:, :, t],
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles[:, t],
            D=d,
            Z=z[:, t],
            Input_States=states,
        )
        decode_chunks.append(out_t.unsqueeze(1))

    if decode_chunks:
        out_decode = torch.cat(decode_chunks, dim=1)
        out_rollout = torch.cat([out_prefill, out_decode], dim=1)
    else:
        out_rollout = out_prefill

    assert (out_full.float() - out_rollout.float()).abs().max().item() < 7e-2
    for state_rollout, state_full in zip(states, full_states):
        assert (state_rollout.float() - state_full.float()).abs().max().item() < 1e-1
