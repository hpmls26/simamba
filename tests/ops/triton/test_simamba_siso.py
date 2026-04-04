"""Simamba Triton SISO forward parity tests."""

import sys
import types
from pathlib import Path

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
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import simamba_siso_combined


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_simamba_triton_forward_varlen_matches_reference():
    torch.manual_seed(321)
    device = "cuda"

    batch = 1
    nheads_qk = 2
    nheads = 4
    headdim_qk = 8
    headdim_v = 8
    n_angles = 4
    cu_seqlens = torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int32)
    total_seqlen = int(cu_seqlens[-1].item())
    num_sequences = cu_seqlens.numel() - 1

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

    init_states = (
        torch.randn(num_sequences, nheads, n_angles, device=device, dtype=torch.float32),
        torch.randn(num_sequences, nheads, headdim_v, headdim_qk, device=device, dtype=torch.float32),
        torch.randn(num_sequences, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(num_sequences, nheads, headdim_qk, device=device, dtype=torch.bfloat16),
        torch.randn(num_sequences, nheads, headdim_v, device=device, dtype=torch.bfloat16),
        torch.randn(num_sequences, nheads, headdim_v, device=device, dtype=torch.bfloat16),
    )

    ref_out = torch.empty((batch, total_seqlen, nheads, headdim_v), device=device, dtype=torch.bfloat16)
    ref_final = [[], [], [], [], [], []]
    for seq_idx in range(num_sequences):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        seq_states = tuple(s[seq_idx : seq_idx + 1] for s in init_states)
        ref = simamba_siso_combined(
            Q=q[:, start:end],
            K=k[:, start:end],
            V=v[:, start:end],
            ADT=adt[:, :, start:end],
            DT=dt[:, :, start:end],
            Simpson=simpson[:, :, start:end],
            Midpoint=midpoint[:, :, start:end],
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles[:, start:end],
            D=d,
            Z=z[:, start:end],
            Input_States=seq_states,
            return_final_states=True,
        )
        ref_out[:, start:end] = ref[0]
        for i in range(6):
            ref_final[i].append(ref[1 + i])

    ref_final = tuple(torch.cat(parts, dim=0) for parts in ref_final)

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
        Initial_States=init_states,
        return_final_states=True,
        cu_seqlens=cu_seqlens,
    )

    out_tri = tri[0]
    final_tri = tri[-1]
    assert final_tri is not None

    assert (ref_out.float() - out_tri.float()).abs().max().item() < 6e-2
    assert (ref_final[0].float() - final_tri[0].float()).abs().max().item() < 1e-2
    assert (ref_final[1].float() - final_tri[1].float()).abs().max().item() < 9e-2
    assert (ref_final[2].float() - final_tri[2].float()).abs().max().item() < 6e-2
    assert (ref_final[3].float() - final_tri[3].float()).abs().max().item() < 6e-2
    assert (ref_final[4].float() - final_tri[4].float()).abs().max().item() < 6e-2
    assert (ref_final[5].float() - final_tri[5].float()).abs().max().item() < 6e-2


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
