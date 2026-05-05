"""Simamba Triton backward helpers for Simpson coefficients.

Phase 6 scope in this file:
- dADT, dDT, dSimpson, and optional dMidpoint.
- Reverse-time recursion over SSM state.
- Per-step coefficient gradients computed by a Triton kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from simamba_siso_combined import (
    _apply_pairwise_rotary,
    _compute_angle_cumsum,
    _resolve_simpson_effective,
    _simpson_state_update,
    simamba_siso_combined,
)


@triton.jit
def _simamba_coeff_bwd_step_kernel(
    DSSM,
    SSM_PREV,
    KV_T,
    KV_PREV1,
    KV_PREV2,
    ADT,
    DT,
    S_EFF,
    DADT,
    DDT,
    DSEFF,
    stride_dssm_batch,
    stride_dssm_head,
    stride_dssm_v,
    stride_dssm_qk,
    stride_ssm_prev_batch,
    stride_ssm_prev_head,
    stride_ssm_prev_v,
    stride_ssm_prev_qk,
    stride_kv_t_batch,
    stride_kv_t_head,
    stride_kv_t_v,
    stride_kv_t_qk,
    stride_kv_prev1_batch,
    stride_kv_prev1_head,
    stride_kv_prev1_v,
    stride_kv_prev1_qk,
    stride_kv_prev2_batch,
    stride_kv_prev2_head,
    stride_kv_prev2_v,
    stride_kv_prev2_qk,
    stride_adt_batch,
    stride_adt_head,
    stride_dt_batch,
    stride_dt_head,
    stride_s_eff_batch,
    stride_s_eff_head,
    stride_dadt_batch,
    stride_dadt_head,
    stride_ddt_batch,
    stride_ddt_head,
    stride_dseff_batch,
    stride_dseff_head,
    headdim_v,
    headdim_qk,
    BLOCK_V: tl.constexpr,
    BLOCK_QK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_v = tl.arange(0, BLOCK_V)
    offs_qk = tl.arange(0, BLOCK_QK)
    mask = (offs_v[:, None] < headdim_v) & (offs_qk[None, :] < headdim_qk)

    dssm_ptr = (
        DSSM
        + pid_b * stride_dssm_batch
        + pid_h * stride_dssm_head
        + offs_v[:, None] * stride_dssm_v
        + offs_qk[None, :] * stride_dssm_qk
    )
    ssm_prev_ptr = (
        SSM_PREV
        + pid_b * stride_ssm_prev_batch
        + pid_h * stride_ssm_prev_head
        + offs_v[:, None] * stride_ssm_prev_v
        + offs_qk[None, :] * stride_ssm_prev_qk
    )
    kv_t_ptr = (
        KV_T
        + pid_b * stride_kv_t_batch
        + pid_h * stride_kv_t_head
        + offs_v[:, None] * stride_kv_t_v
        + offs_qk[None, :] * stride_kv_t_qk
    )
    kv_prev1_ptr = (
        KV_PREV1
        + pid_b * stride_kv_prev1_batch
        + pid_h * stride_kv_prev1_head
        + offs_v[:, None] * stride_kv_prev1_v
        + offs_qk[None, :] * stride_kv_prev1_qk
    )
    kv_prev2_ptr = (
        KV_PREV2
        + pid_b * stride_kv_prev2_batch
        + pid_h * stride_kv_prev2_head
        + offs_v[:, None] * stride_kv_prev2_v
        + offs_qk[None, :] * stride_kv_prev2_qk
    )

    dssm = tl.load(dssm_ptr, mask=mask, other=0.0).to(tl.float32)
    ssm_prev = tl.load(ssm_prev_ptr, mask=mask, other=0.0).to(tl.float32)
    kv_t = tl.load(kv_t_ptr, mask=mask, other=0.0).to(tl.float32)
    kv_prev1 = tl.load(kv_prev1_ptr, mask=mask, other=0.0).to(tl.float32)
    kv_prev2 = tl.load(kv_prev2_ptr, mask=mask, other=0.0).to(tl.float32)

    ch = tl.sum(tl.sum(dssm * ssm_prev, axis=1), axis=0)
    c0 = tl.sum(tl.sum(dssm * kv_t, axis=1), axis=0)
    c1 = tl.sum(tl.sum(dssm * kv_prev1, axis=1), axis=0)
    c2 = tl.sum(tl.sum(dssm * kv_prev2, axis=1), axis=0)

    adt = tl.load(ADT + pid_b * stride_adt_batch + pid_h * stride_adt_head).to(tl.float32)
    dt = tl.load(DT + pid_b * stride_dt_batch + pid_h * stride_dt_head).to(tl.float32)
    s_eff = tl.load(S_EFF + pid_b * stride_s_eff_batch + pid_h * stride_s_eff_head).to(tl.float32)

    alpha = tl.math.exp2(adt * 1.44269504089)
    alpha_half = tl.math.exp2(adt * 0.721347520445)

    dadt = (
        alpha * ch
        + (dt / 12.0) * alpha_half * (2.0 - 0.5 * s_eff) * c0
        + ((dt / 6.0) * alpha + (dt / 12.0) * alpha_half * (2.0 + s_eff)) * c1
        + (-(dt / 24.0) * alpha_half * s_eff) * c2
    )
    ddt = (
        (1.0 / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * s_eff)) * c0
        + (1.0 / 6.0) * (alpha + alpha_half * (2.0 + s_eff)) * c1
        + (-(1.0 / 12.0) * alpha_half * s_eff) * c2
    )
    dseff = (
        (-(dt / 12.0) * alpha_half) * c0
        + ((dt / 6.0) * alpha_half) * c1
        + (-(dt / 12.0) * alpha_half) * c2
    )

    tl.store(DADT + pid_b * stride_dadt_batch + pid_h * stride_dadt_head, dadt)
    tl.store(DDT + pid_b * stride_ddt_batch + pid_h * stride_ddt_head, ddt)
    tl.store(DSEFF + pid_b * stride_dseff_batch + pid_h * stride_dseff_head, dseff)


def _compute_step_coeff_grads_triton(
    dssm: torch.Tensor,
    ssm_prev: torch.Tensor,
    kv_t: torch.Tensor,
    kv_prev1: torch.Tensor,
    kv_prev2: torch.Tensor,
    adt_t: torch.Tensor,
    dt_t: torch.Tensor,
    s_eff_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-step coefficient grads with Triton."""
    batch, nheads, headdim_v, headdim_qk = dssm.shape

    if not dssm.is_cuda:
        raise ValueError("Triton coefficient backward requires CUDA tensors.")

    dssm = dssm.contiguous()
    ssm_prev = ssm_prev.contiguous()
    kv_t = kv_t.contiguous()
    kv_prev1 = kv_prev1.contiguous()
    kv_prev2 = kv_prev2.contiguous()
    adt_t = adt_t.contiguous()
    dt_t = dt_t.contiguous()
    s_eff_t = s_eff_t.contiguous()

    dadt_t = torch.empty((batch, nheads), device=dssm.device, dtype=torch.float32)
    ddt_t = torch.empty((batch, nheads), device=dssm.device, dtype=torch.float32)
    dseff_t = torch.empty((batch, nheads), device=dssm.device, dtype=torch.float32)

    block_v = triton.next_power_of_2(headdim_v)
    block_qk = triton.next_power_of_2(headdim_qk)
    grid = (nheads, batch)

    _simamba_coeff_bwd_step_kernel[grid](
        dssm,
        ssm_prev,
        kv_t,
        kv_prev1,
        kv_prev2,
        adt_t,
        dt_t,
        s_eff_t,
        dadt_t,
        ddt_t,
        dseff_t,
        dssm.stride(0),
        dssm.stride(1),
        dssm.stride(2),
        dssm.stride(3),
        ssm_prev.stride(0),
        ssm_prev.stride(1),
        ssm_prev.stride(2),
        ssm_prev.stride(3),
        kv_t.stride(0),
        kv_t.stride(1),
        kv_t.stride(2),
        kv_t.stride(3),
        kv_prev1.stride(0),
        kv_prev1.stride(1),
        kv_prev1.stride(2),
        kv_prev1.stride(3),
        kv_prev2.stride(0),
        kv_prev2.stride(1),
        kv_prev2.stride(2),
        kv_prev2.stride(3),
        adt_t.stride(0),
        adt_t.stride(1),
        dt_t.stride(0),
        dt_t.stride(1),
        s_eff_t.stride(0),
        s_eff_t.stride(1),
        dadt_t.stride(0),
        dadt_t.stride(1),
        ddt_t.stride(0),
        ddt_t.stride(1),
        dseff_t.stride(0),
        dseff_t.stride(1),
        headdim_v,
        headdim_qk,
        BLOCK_V=block_v,
        BLOCK_QK=block_qk,
    )

    return dadt_t, ddt_t, dseff_t


def compute_dcoeffs(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    grad_out: torch.Tensor,
    Midpoint: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute Simamba coefficient gradients.

    Notes:
    - dADT/dSimpson/dMidpoint are computed from the explicit Simpson recurrence.
    - dDT receives an additional reference autograd correction so DT gradients
      include the DT->rotary-angle path from the forward pass.
    """

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape

    if grad_out.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(
            f"grad_out shape mismatch: expected {(batch, seqlen, nheads, headdim_v)}, got {grad_out.shape}."
        )

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa_ratio, dim=2)
        K = K.repeat_interleave(gqa_ratio, dim=2)

    n_angles = Angles.shape[-1]
    if Input_States is not None:
        (
            input_angle_state,
            ssm_state,
            k_prev1,
            k_prev2,
            v_prev1,
            v_prev2,
        ) = Input_States
    else:
        input_angle_state = None
        ssm_state = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
        k_prev1 = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        k_prev2 = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        v_prev1 = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)
        v_prev2 = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)

    angles_cumsum, _ = _compute_angle_cumsum(Angles, DT, input_angle_state)
    q_pre = Q + Q_bias[None, None, :, :]
    k_pre = K + K_bias[None, None, :, :]
    q_rot = _apply_pairwise_rotary(q_pre, angles_cumsum).float()
    k_rot = _apply_pairwise_rotary(k_pre, angles_cumsum).float()

    ssm_prev_seq = torch.empty((batch, seqlen, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    kv_t_seq = torch.empty_like(ssm_prev_seq)
    kv_prev1_seq = torch.empty_like(ssm_prev_seq)
    kv_prev2_seq = torch.empty_like(ssm_prev_seq)
    s_eff = torch.empty((batch, nheads, seqlen), device=Q.device, dtype=torch.float32)

    ssm_running = ssm_state.float()
    k_prev1_running = k_prev1.float()
    k_prev2_running = k_prev2.float()
    v_prev1_running = v_prev1.float()
    v_prev2_running = v_prev2.float()

    for t in range(seqlen):
        ssm_prev_seq[:, t] = ssm_running

        v_t = V[:, t].float()
        k_t = k_rot[:, t].float()

        kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        kv_prev1 = v_prev1_running.unsqueeze(-1) * k_prev1_running.unsqueeze(-2)
        kv_prev2 = v_prev2_running.unsqueeze(-1) * k_prev2_running.unsqueeze(-2)

        kv_t_seq[:, t] = kv_t
        kv_prev1_seq[:, t] = kv_prev1
        kv_prev2_seq[:, t] = kv_prev2

        midpoint_t = Midpoint[:, :, t] if Midpoint is not None else None
        s_eff[:, :, t] = _resolve_simpson_effective(Simpson[:, :, t], midpoint_t)

        ssm_running = _simpson_state_update(
            ssm_running,
            kv_t,
            kv_prev1,
            kv_prev2,
            ADT[:, :, t],
            DT[:, :, t],
            Simpson[:, :, t],
            midpoint_t,
        )

        k_prev2_running = k_prev1_running
        k_prev1_running = k_t
        v_prev2_running = v_prev1_running
        v_prev1_running = v_t

    if Z is not None:
        grad_eff = grad_out.float() * F.silu(Z.float())
    else:
        grad_eff = grad_out.float()

    dssm_from_out = grad_eff.unsqueeze(-1) * q_rot.unsqueeze(-2)

    dadt = torch.empty_like(ADT, dtype=torch.float32)
    ddt = torch.empty_like(DT, dtype=torch.float32)
    dseff = torch.empty_like(Simpson, dtype=torch.float32)

    dssm_future = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    alpha = torch.exp(ADT.float())

    for t in range(seqlen - 1, -1, -1):
        dssm_t = dssm_from_out[:, t] + dssm_future
        dadt_t, ddt_t, dseff_t = _compute_step_coeff_grads_triton(
            dssm=dssm_t,
            ssm_prev=ssm_prev_seq[:, t],
            kv_t=kv_t_seq[:, t],
            kv_prev1=kv_prev1_seq[:, t],
            kv_prev2=kv_prev2_seq[:, t],
            adt_t=ADT[:, :, t].float(),
            dt_t=DT[:, :, t].float(),
            s_eff_t=s_eff[:, :, t],
        )

        dadt[:, :, t] = dadt_t
        ddt[:, :, t] = ddt_t
        dseff[:, :, t] = dseff_t

        dssm_future = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1) * dssm_t

    if Midpoint is None:
        dsimpson = dseff
        dmidpoint = None
    else:
        dsimpson = dseff * Midpoint.float()
        dmidpoint = dseff * Simpson.float()

    # DT also influences rotary angle accumulation in forward. Add that path via
    # reference autograd so finite-difference checks match end-to-end behavior.
    dt_ref = DT.detach().clone().requires_grad_(True)
    out_ref = simamba_siso_combined(
        Q=Q.detach(),
        K=K.detach(),
        V=V.detach(),
        ADT=ADT.detach(),
        DT=dt_ref,
        Simpson=Simpson.detach(),
        Midpoint=Midpoint.detach() if Midpoint is not None else None,
        Q_bias=Q_bias.detach(),
        K_bias=K_bias.detach(),
        Angles=Angles.detach(),
        D=D.detach() if D is not None else None,
        Z=Z.detach() if Z is not None else None,
        Input_States=tuple(s.detach() for s in Input_States) if Input_States is not None else None,
        return_final_states=False,
    )
    dt_ref_loss = (out_ref.float() * grad_out.float()).sum()
    ddt = torch.autograd.grad(dt_ref_loss, dt_ref, retain_graph=False, create_graph=False)[0]

    return dadt, ddt, dsimpson, dmidpoint
