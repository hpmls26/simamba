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

from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (
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


@triton.jit
def _simamba_main_bwd_kernel(
    Q,
    K,
    V,
    ADT,
    DT,
    Simpson,
    Midpoint,
    Q_bias,
    K_bias,
    Angles_Cumsum,
    D,
    DO,
    Chunk_Start_SSM_State,
    Chunk_Start_K_Prev1_State,
    Chunk_Start_K_Prev2_State,
    Chunk_Start_V_Prev1_State,
    Chunk_Start_V_Prev2_State,
    Final_SSM_State,
    Grad_Final_SSM_State,
    Grad_Final_K_Prev1_State,
    Grad_Final_K_Prev2_State,
    Grad_Final_V_Prev1_State,
    Grad_Final_V_Prev2_State,
    dQ_partial,
    dK_partial,
    dV,
    dADT,
    dDT,
    dSimpson,
    dMidpoint,
    dAngles_Cumsum,
    dQ_bias_partial,
    dK_bias_partial,
    dD_partial,
    dInput_SSM_State,
    dInput_K_Prev1_State,
    dInput_K_Prev2_State,
    dInput_V_Prev1_State,
    dInput_V_Prev2_State,
    stride_q_batch,
    stride_q_seqlen,
    stride_q_head,
    stride_q_qkdim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_head,
    stride_k_qkdim,
    stride_v_batch,
    stride_v_seqlen,
    stride_v_head,
    stride_v_vdim,
    stride_adt_batch,
    stride_adt_head,
    stride_adt_seqlen,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_seqlen,
    stride_simpson_batch,
    stride_simpson_head,
    stride_simpson_seqlen,
    stride_midpoint_batch,
    stride_midpoint_head,
    stride_midpoint_seqlen,
    stride_q_bias_head,
    stride_q_bias_qkdim,
    stride_k_bias_head,
    stride_k_bias_qkdim,
    stride_angles_batch,
    stride_angles_seqlen,
    stride_angles_head,
    stride_angles_qkdim,
    stride_d_head,
    stride_do_batch,
    stride_do_seqlen,
    stride_do_head,
    stride_do_vdim,
    stride_chunk_ssm_batch,
    stride_chunk_ssm_head,
    stride_chunk_ssm_chunk,
    stride_chunk_ssm_vdim,
    stride_chunk_ssm_qkdim,
    stride_chunk_k1_batch,
    stride_chunk_k1_head,
    stride_chunk_k1_chunk,
    stride_chunk_k1_dim,
    stride_chunk_k2_batch,
    stride_chunk_k2_head,
    stride_chunk_k2_chunk,
    stride_chunk_k2_dim,
    stride_chunk_v1_batch,
    stride_chunk_v1_head,
    stride_chunk_v1_chunk,
    stride_chunk_v1_dim,
    stride_chunk_v2_batch,
    stride_chunk_v2_head,
    stride_chunk_v2_chunk,
    stride_chunk_v2_dim,
    stride_final_ssm_batch,
    stride_final_ssm_head,
    stride_final_ssm_vdim,
    stride_final_ssm_qkdim,
    stride_grad_final_ssm_batch,
    stride_grad_final_ssm_head,
    stride_grad_final_ssm_vdim,
    stride_grad_final_ssm_qkdim,
    stride_grad_final_k1_batch,
    stride_grad_final_k1_head,
    stride_grad_final_k1_dim,
    stride_grad_final_k2_batch,
    stride_grad_final_k2_head,
    stride_grad_final_k2_dim,
    stride_grad_final_v1_batch,
    stride_grad_final_v1_head,
    stride_grad_final_v1_dim,
    stride_grad_final_v2_batch,
    stride_grad_final_v2_head,
    stride_grad_final_v2_dim,
    stride_dq_partial_batch,
    stride_dq_partial_seqlen,
    stride_dq_partial_head,
    stride_dq_partial_qkdim,
    stride_dk_partial_batch,
    stride_dk_partial_seqlen,
    stride_dk_partial_head,
    stride_dk_partial_qkdim,
    stride_dv_batch,
    stride_dv_seqlen,
    stride_dv_head,
    stride_dv_vdim,
    stride_dadt_batch,
    stride_dadt_head,
    stride_dadt_seqlen,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_seqlen,
    stride_dsimpson_batch,
    stride_dsimpson_head,
    stride_dsimpson_seqlen,
    stride_dmidpoint_batch,
    stride_dmidpoint_head,
    stride_dmidpoint_seqlen,
    stride_dangles_batch,
    stride_dangles_seqlen,
    stride_dangles_head,
    stride_dangles_qkdim,
    stride_dq_bias_partial_batch,
    stride_dq_bias_partial_head,
    stride_dq_bias_partial_qkdim,
    stride_dk_bias_partial_batch,
    stride_dk_bias_partial_head,
    stride_dk_bias_partial_qkdim,
    stride_dd_partial_batch,
    stride_dd_partial_head,
    stride_dinput_ssm_batch,
    stride_dinput_ssm_head,
    stride_dinput_ssm_vdim,
    stride_dinput_ssm_qkdim,
    stride_dinput_k1_batch,
    stride_dinput_k1_head,
    stride_dinput_k1_dim,
    stride_dinput_k2_batch,
    stride_dinput_k2_head,
    stride_dinput_k2_dim,
    stride_dinput_v1_batch,
    stride_dinput_v1_head,
    stride_dinput_v1_dim,
    stride_dinput_v2_batch,
    stride_dinput_v2_head,
    stride_dinput_v2_dim,
    seqlen,
    nheads_qk,
    headdim_qk,
    headdim_v,
    headdim_angles,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_MIDPOINT: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    gqa_ratio = nheads // nheads_qk
    head_idx_qk = pid_head // gqa_ratio

    offs_qk = tl.arange(0, HEADDIM_QK)
    offs_v = tl.arange(0, HEADDIM_V)
    offs_angle = tl.arange(0, HEADDIM_QK // 2)
    qk_mask = offs_qk < headdim_qk
    v_mask = offs_v < headdim_v
    angle_mask = offs_angle < headdim_angles

    dq_bias_acc = tl.zeros([HEADDIM_QK], dtype=tl.float32)
    dk_bias_acc = tl.zeros([HEADDIM_QK], dtype=tl.float32)
    dD_acc = tl.zeros([1], dtype=tl.float32)

    dssm_carry = tl.load(
        Grad_Final_SSM_State
        + pid_batch * stride_grad_final_ssm_batch
        + pid_head * stride_grad_final_ssm_head
        + offs_v[:, None] * stride_grad_final_ssm_vdim
        + offs_qk[None, :] * stride_grad_final_ssm_qkdim,
        mask=v_mask[:, None] & qk_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    end_k_prev1_grad = tl.load(
        Grad_Final_K_Prev1_State
        + pid_batch * stride_grad_final_k1_batch
        + pid_head * stride_grad_final_k1_head
        + offs_qk * stride_grad_final_k1_dim,
        mask=qk_mask,
        other=0.0,
    ).to(tl.float32)
    end_k_prev2_grad = tl.load(
        Grad_Final_K_Prev2_State
        + pid_batch * stride_grad_final_k2_batch
        + pid_head * stride_grad_final_k2_head
        + offs_qk * stride_grad_final_k2_dim,
        mask=qk_mask,
        other=0.0,
    ).to(tl.float32)
    end_v_prev1_grad = tl.load(
        Grad_Final_V_Prev1_State
        + pid_batch * stride_grad_final_v1_batch
        + pid_head * stride_grad_final_v1_head
        + offs_v * stride_grad_final_v1_dim,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)
    end_v_prev2_grad = tl.load(
        Grad_Final_V_Prev2_State
        + pid_batch * stride_grad_final_v2_batch
        + pid_head * stride_grad_final_v2_head
        + offs_v * stride_grad_final_v2_dim,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    dkv_carry1 = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)
    dkv_carry2 = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    q_bias = tl.load(
        Q_bias + pid_head * stride_q_bias_head + offs_qk * stride_q_bias_qkdim,
        mask=qk_mask,
        other=0.0,
    ).to(tl.float32)
    k_bias = tl.load(
        K_bias + pid_head * stride_k_bias_head + offs_qk * stride_k_bias_qkdim,
        mask=qk_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_D:
        D_val = tl.load(D + pid_head * stride_d_head).to(tl.float32)

    for chunk_idx in range(num_chunks - 1, -1, -1):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_len = tl.minimum(CHUNK_SIZE, seqlen - chunk_start)

        start_ssm = tl.load(
            Chunk_Start_SSM_State
            + pid_batch * stride_chunk_ssm_batch
            + pid_head * stride_chunk_ssm_head
            + chunk_idx * stride_chunk_ssm_chunk
            + offs_v[:, None] * stride_chunk_ssm_vdim
            + offs_qk[None, :] * stride_chunk_ssm_qkdim,
            mask=v_mask[:, None] & qk_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        start_k_prev1 = tl.load(
            Chunk_Start_K_Prev1_State
            + pid_batch * stride_chunk_k1_batch
            + pid_head * stride_chunk_k1_head
            + chunk_idx * stride_chunk_k1_chunk
            + offs_qk * stride_chunk_k1_dim,
            mask=qk_mask,
            other=0.0,
        ).to(tl.float32)
        start_k_prev2 = tl.load(
            Chunk_Start_K_Prev2_State
            + pid_batch * stride_chunk_k2_batch
            + pid_head * stride_chunk_k2_head
            + chunk_idx * stride_chunk_k2_chunk
            + offs_qk * stride_chunk_k2_dim,
            mask=qk_mask,
            other=0.0,
        ).to(tl.float32)
        start_v_prev1 = tl.load(
            Chunk_Start_V_Prev1_State
            + pid_batch * stride_chunk_v1_batch
            + pid_head * stride_chunk_v1_head
            + chunk_idx * stride_chunk_v1_chunk
            + offs_v * stride_chunk_v1_dim,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)
        start_v_prev2 = tl.load(
            Chunk_Start_V_Prev2_State
            + pid_batch * stride_chunk_v2_batch
            + pid_head * stride_chunk_v2_head
            + chunk_idx * stride_chunk_v2_chunk
            + offs_v * stride_chunk_v2_dim,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        if chunk_idx == num_chunks - 1:
            h_current = tl.load(
                Final_SSM_State
                + pid_batch * stride_final_ssm_batch
                + pid_head * stride_final_ssm_head
                + offs_v[:, None] * stride_final_ssm_vdim
                + offs_qk[None, :] * stride_final_ssm_qkdim,
                mask=v_mask[:, None] & qk_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            h_current = tl.load(
                Chunk_Start_SSM_State
                + pid_batch * stride_chunk_ssm_batch
                + pid_head * stride_chunk_ssm_head
                + (chunk_idx + 1) * stride_chunk_ssm_chunk
                + offs_v[:, None] * stride_chunk_ssm_vdim
                + offs_qk[None, :] * stride_chunk_ssm_qkdim,
                mask=v_mask[:, None] & qk_mask[None, :],
                other=0.0,
            ).to(tl.float32)

        k_extra_cur = end_k_prev1_grad
        v_extra_cur = end_v_prev1_grad
        k_extra_next = tl.zeros([HEADDIM_QK], dtype=tl.float32)
        v_extra_next = tl.zeros([HEADDIM_V], dtype=tl.float32)
        boundary_k_prev1_extra = tl.zeros([HEADDIM_QK], dtype=tl.float32)
        boundary_v_prev1_extra = tl.zeros([HEADDIM_V], dtype=tl.float32)
        if chunk_len > 1:
            k_extra_next = end_k_prev2_grad
            v_extra_next = end_v_prev2_grad
        else:
            boundary_k_prev1_extra = end_k_prev2_grad
            boundary_v_prev1_extra = end_v_prev2_grad

        for rel_t in range(CHUNK_SIZE - 1, -1, -1):
            if rel_t < chunk_len:
                global_t = chunk_start + rel_t

                q_pre = tl.load(
                    Q
                    + pid_batch * stride_q_batch
                    + global_t * stride_q_seqlen
                    + head_idx_qk * stride_q_head
                    + offs_qk * stride_q_qkdim,
                    mask=qk_mask,
                    other=0.0,
                ).to(tl.float32) + q_bias
                k_pre = tl.load(
                    K
                    + pid_batch * stride_k_batch
                    + global_t * stride_k_seqlen
                    + head_idx_qk * stride_k_head
                    + offs_qk * stride_k_qkdim,
                    mask=qk_mask,
                    other=0.0,
                ).to(tl.float32) + k_bias
                v_t = tl.load(
                    V
                    + pid_batch * stride_v_batch
                    + global_t * stride_v_seqlen
                    + pid_head * stride_v_head
                    + offs_v * stride_v_vdim,
                    mask=v_mask,
                    other=0.0,
                ).to(tl.float32)

                theta = tl.load(
                    Angles_Cumsum
                    + pid_batch * stride_angles_batch
                    + global_t * stride_angles_seqlen
                    + pid_head * stride_angles_head
                    + offs_angle * stride_angles_qkdim,
                    mask=angle_mask,
                    other=0.0,
                ).to(tl.float32)
                cos_theta = cos_approx(theta)
                sin_theta = sin_approx(theta)

                q_pairs = tl.reshape(q_pre, [HEADDIM_QK // 2, 2])
                q0, q1 = tl.split(q_pairs)
                q_rot0 = q0 * cos_theta - q1 * sin_theta
                q_rot1 = q0 * sin_theta + q1 * cos_theta
                q_rot = tl.reshape(tl.join(q_rot0, q_rot1), [HEADDIM_QK])

                k_pairs = tl.reshape(k_pre, [HEADDIM_QK // 2, 2])
                k0, k1 = tl.split(k_pairs)
                k_rot0 = k0 * cos_theta - k1 * sin_theta
                k_rot1 = k0 * sin_theta + k1 * cos_theta
                k_rot = tl.reshape(tl.join(k_rot0, k_rot1), [HEADDIM_QK])

                if rel_t > 0:
                    prev1_t = global_t - 1
                    k_prev1_pre = tl.load(
                        K
                        + pid_batch * stride_k_batch
                        + prev1_t * stride_k_seqlen
                        + head_idx_qk * stride_k_head
                        + offs_qk * stride_k_qkdim,
                        mask=qk_mask,
                        other=0.0,
                    ).to(tl.float32) + k_bias
                    theta_prev1 = tl.load(
                        Angles_Cumsum
                        + pid_batch * stride_angles_batch
                        + prev1_t * stride_angles_seqlen
                        + pid_head * stride_angles_head
                        + offs_angle * stride_angles_qkdim,
                        mask=angle_mask,
                        other=0.0,
                    ).to(tl.float32)
                    cos_prev1 = cos_approx(theta_prev1)
                    sin_prev1 = sin_approx(theta_prev1)
                    k_prev1_pairs = tl.reshape(k_prev1_pre, [HEADDIM_QK // 2, 2])
                    kp10, kp11 = tl.split(k_prev1_pairs)
                    k_prev1_rot0 = kp10 * cos_prev1 - kp11 * sin_prev1
                    k_prev1_rot1 = kp10 * sin_prev1 + kp11 * cos_prev1
                    k_prev1 = tl.reshape(tl.join(k_prev1_rot0, k_prev1_rot1), [HEADDIM_QK])
                    v_prev1 = tl.load(
                        V
                        + pid_batch * stride_v_batch
                        + prev1_t * stride_v_seqlen
                        + pid_head * stride_v_head
                        + offs_v * stride_v_vdim,
                        mask=v_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    k_prev1 = start_k_prev1
                    v_prev1 = start_v_prev1

                if rel_t > 1:
                    prev2_t = global_t - 2
                    k_prev2_pre = tl.load(
                        K
                        + pid_batch * stride_k_batch
                        + prev2_t * stride_k_seqlen
                        + head_idx_qk * stride_k_head
                        + offs_qk * stride_k_qkdim,
                        mask=qk_mask,
                        other=0.0,
                    ).to(tl.float32) + k_bias
                    theta_prev2 = tl.load(
                        Angles_Cumsum
                        + pid_batch * stride_angles_batch
                        + prev2_t * stride_angles_seqlen
                        + pid_head * stride_angles_head
                        + offs_angle * stride_angles_qkdim,
                        mask=angle_mask,
                        other=0.0,
                    ).to(tl.float32)
                    cos_prev2 = cos_approx(theta_prev2)
                    sin_prev2 = sin_approx(theta_prev2)
                    k_prev2_pairs = tl.reshape(k_prev2_pre, [HEADDIM_QK // 2, 2])
                    kp20, kp21 = tl.split(k_prev2_pairs)
                    k_prev2_rot0 = kp20 * cos_prev2 - kp21 * sin_prev2
                    k_prev2_rot1 = kp20 * sin_prev2 + kp21 * cos_prev2
                    k_prev2 = tl.reshape(tl.join(k_prev2_rot0, k_prev2_rot1), [HEADDIM_QK])
                    v_prev2 = tl.load(
                        V
                        + pid_batch * stride_v_batch
                        + prev2_t * stride_v_seqlen
                        + pid_head * stride_v_head
                        + offs_v * stride_v_vdim,
                        mask=v_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    k_prev2 = start_k_prev2
                    v_prev2 = start_v_prev2

                adt_t = tl.load(
                    ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head + global_t * stride_adt_seqlen
                ).to(tl.float32)
                dt_t = tl.load(
                    DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head + global_t * stride_dt_seqlen
                ).to(tl.float32)
                simpson_t = tl.load(
                    Simpson
                    + pid_batch * stride_simpson_batch
                    + pid_head * stride_simpson_head
                    + global_t * stride_simpson_seqlen
                ).to(tl.float32)
                simpson_t = tl.maximum(0.0, tl.minimum(1.0, simpson_t))
                if HAS_MIDPOINT:
                    midpoint_t = tl.load(
                        Midpoint
                        + pid_batch * stride_midpoint_batch
                        + pid_head * stride_midpoint_head
                        + global_t * stride_midpoint_seqlen
                    ).to(tl.float32)
                    midpoint_t = tl.maximum(0.0, tl.minimum(1.0, midpoint_t))
                    s_eff_t = tl.maximum(0.0, tl.minimum(1.0, simpson_t * midpoint_t))
                else:
                    midpoint_t = 0.0
                    s_eff_t = simpson_t

                alpha = tl.math.exp2(adt_t * 1.44269504089)
                alpha_half = tl.math.exp2(adt_t * 0.721347520445)
                gamma0 = (dt_t / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * s_eff_t))
                gamma1 = (dt_t / 6.0) * (alpha + alpha_half * (2.0 + s_eff_t))
                gamma2 = -(dt_t / 12.0) * alpha_half * s_eff_t

                kv_t = v_t[:, None] * k_rot[None, :]
                kv_prev1 = v_prev1[:, None] * k_prev1[None, :]
                kv_prev2 = v_prev2[:, None] * k_prev2[None, :]

                do_t = tl.load(
                    DO
                    + pid_batch * stride_do_batch
                    + global_t * stride_do_seqlen
                    + pid_head * stride_do_head
                    + offs_v * stride_do_vdim,
                    mask=v_mask,
                    other=0.0,
                ).to(tl.float32)

                dssm_total = dssm_carry + do_t[:, None] * q_rot[None, :]
                dQ_rot = tl.sum(h_current * do_t[:, None], axis=0)
                dkv_total = dkv_carry1 + gamma0 * dssm_total

                dV_t = tl.sum(dkv_total * k_rot[None, :], axis=1) + v_extra_cur
                dK_rot = tl.sum(dkv_total * v_t[:, None], axis=0) + k_extra_cur
                if HAS_D:
                    dV_t += D_val * do_t
                    dD_acc += tl.sum(do_t * v_t, axis=0)

                h_prev = (h_current - gamma0 * kv_t - gamma1 * kv_prev1 - gamma2 * kv_prev2) / alpha

                ch = tl.sum(dssm_total * h_prev, axis=0)
                ch = tl.sum(ch, axis=0)
                c0 = tl.sum(dssm_total * kv_t, axis=0)
                c0 = tl.sum(c0, axis=0)
                c1 = tl.sum(dssm_total * kv_prev1, axis=0)
                c1 = tl.sum(c1, axis=0)
                c2 = tl.sum(dssm_total * kv_prev2, axis=0)
                c2 = tl.sum(c2, axis=0)

                dadt_t = (
                    alpha * ch
                    + (dt_t / 12.0) * alpha_half * (2.0 - 0.5 * s_eff_t) * c0
                    + ((dt_t / 6.0) * alpha + (dt_t / 12.0) * alpha_half * (2.0 + s_eff_t)) * c1
                    + (-(dt_t / 24.0) * alpha_half * s_eff_t) * c2
                )
                ddt_t = (
                    (1.0 / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * s_eff_t)) * c0
                    + (1.0 / 6.0) * (alpha + alpha_half * (2.0 + s_eff_t)) * c1
                    + (-(1.0 / 12.0) * alpha_half * s_eff_t) * c2
                )
                dseff_t = (
                    (-(dt_t / 12.0) * alpha_half) * c0
                    + ((dt_t / 6.0) * alpha_half) * c1
                    + (-(dt_t / 12.0) * alpha_half) * c2
                )

                dQ_rot_pairs = tl.reshape(dQ_rot, [HEADDIM_QK // 2, 2])
                dqr0, dqr1 = tl.split(dQ_rot_pairs)
                dK_rot_pairs = tl.reshape(dK_rot, [HEADDIM_QK // 2, 2])
                dkr0, dkr1 = tl.split(dK_rot_pairs)

                dq0 = dqr0 * cos_theta + dqr1 * sin_theta
                dq1 = -dqr0 * sin_theta + dqr1 * cos_theta
                dk0 = dkr0 * cos_theta + dkr1 * sin_theta
                dk1 = -dkr0 * sin_theta + dkr1 * cos_theta

                dQ_pre = tl.reshape(tl.join(dq0, dq1), [HEADDIM_QK])
                dK_pre = tl.reshape(tl.join(dk0, dk1), [HEADDIM_QK])
                dq_bias_acc += dQ_pre
                dk_bias_acc += dK_pre

                dtheta_q = dqr0 * (-q0 * sin_theta - q1 * cos_theta) + dqr1 * (q0 * cos_theta - q1 * sin_theta)
                dtheta_k = dkr0 * (-k0 * sin_theta - k1 * cos_theta) + dkr1 * (k0 * cos_theta - k1 * sin_theta)
                dtheta = dtheta_q + dtheta_k

                tl.store(
                    dQ_partial
                    + pid_batch * stride_dq_partial_batch
                    + global_t * stride_dq_partial_seqlen
                    + pid_head * stride_dq_partial_head
                    + offs_qk * stride_dq_partial_qkdim,
                    dQ_pre,
                    mask=qk_mask,
                )
                tl.store(
                    dK_partial
                    + pid_batch * stride_dk_partial_batch
                    + global_t * stride_dk_partial_seqlen
                    + pid_head * stride_dk_partial_head
                    + offs_qk * stride_dk_partial_qkdim,
                    dK_pre,
                    mask=qk_mask,
                )
                tl.store(
                    dV
                    + pid_batch * stride_dv_batch
                    + global_t * stride_dv_seqlen
                    + pid_head * stride_dv_head
                    + offs_v * stride_dv_vdim,
                    dV_t,
                    mask=v_mask,
                )
                tl.store(
                    dADT + pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + global_t * stride_dadt_seqlen,
                    dadt_t,
                )
                tl.store(
                    dDT + pid_batch * stride_ddt_batch + pid_head * stride_ddt_head + global_t * stride_ddt_seqlen,
                    ddt_t,
                )
                tl.store(
                    dSimpson
                    + pid_batch * stride_dsimpson_batch
                    + pid_head * stride_dsimpson_head
                    + global_t * stride_dsimpson_seqlen,
                    dseff_t * (midpoint_t if HAS_MIDPOINT else 1.0),
                )
                if HAS_MIDPOINT:
                    tl.store(
                        dMidpoint
                        + pid_batch * stride_dmidpoint_batch
                        + pid_head * stride_dmidpoint_head
                        + global_t * stride_dmidpoint_seqlen,
                        dseff_t * simpson_t,
                    )
                tl.store(
                    dAngles_Cumsum
                    + pid_batch * stride_dangles_batch
                    + global_t * stride_dangles_seqlen
                    + pid_head * stride_dangles_head
                    + offs_angle * stride_dangles_qkdim,
                    dtheta,
                    mask=angle_mask,
                )

                dssm_carry = alpha * dssm_total
                dkv_carry1 = dkv_carry2 + gamma1 * dssm_total
                dkv_carry2 = gamma2 * dssm_total
                h_current = h_prev
                k_extra_cur = k_extra_next
                v_extra_cur = v_extra_next
                k_extra_next = tl.zeros([HEADDIM_QK], dtype=tl.float32)
                v_extra_next = tl.zeros([HEADDIM_V], dtype=tl.float32)

        end_k_prev1_grad = boundary_k_prev1_extra + tl.sum(dkv_carry1 * start_v_prev1[:, None], axis=0)
        end_v_prev1_grad = boundary_v_prev1_extra + tl.sum(dkv_carry1 * start_k_prev1[None, :], axis=1)
        end_k_prev2_grad = tl.sum(dkv_carry2 * start_v_prev2[:, None], axis=0)
        end_v_prev2_grad = tl.sum(dkv_carry2 * start_k_prev2[None, :], axis=1)

    tl.store(
        dQ_bias_partial
        + pid_batch * stride_dq_bias_partial_batch
        + pid_head * stride_dq_bias_partial_head
        + offs_qk * stride_dq_bias_partial_qkdim,
        dq_bias_acc,
        mask=qk_mask,
    )
    tl.store(
        dK_bias_partial
        + pid_batch * stride_dk_bias_partial_batch
        + pid_head * stride_dk_bias_partial_head
        + offs_qk * stride_dk_bias_partial_qkdim,
        dk_bias_acc,
        mask=qk_mask,
    )
    if HAS_D:
        tl.store(
            dD_partial + pid_batch * stride_dd_partial_batch + pid_head * stride_dd_partial_head + tl.arange(0, 1),
            dD_acc,
        )

    tl.store(
        dInput_SSM_State
        + pid_batch * stride_dinput_ssm_batch
        + pid_head * stride_dinput_ssm_head
        + offs_v[:, None] * stride_dinput_ssm_vdim
        + offs_qk[None, :] * stride_dinput_ssm_qkdim,
        dssm_carry,
        mask=v_mask[:, None] & qk_mask[None, :],
    )
    tl.store(
        dInput_K_Prev1_State
        + pid_batch * stride_dinput_k1_batch
        + pid_head * stride_dinput_k1_head
        + offs_qk * stride_dinput_k1_dim,
        end_k_prev1_grad,
        mask=qk_mask,
    )
    tl.store(
        dInput_K_Prev2_State
        + pid_batch * stride_dinput_k2_batch
        + pid_head * stride_dinput_k2_head
        + offs_qk * stride_dinput_k2_dim,
        end_k_prev2_grad,
        mask=qk_mask,
    )
    tl.store(
        dInput_V_Prev1_State
        + pid_batch * stride_dinput_v1_batch
        + pid_head * stride_dinput_v1_head
        + offs_v * stride_dinput_v1_dim,
        end_v_prev1_grad,
        mask=v_mask,
    )
    tl.store(
        dInput_V_Prev2_State
        + pid_batch * stride_dinput_v2_batch
        + pid_head * stride_dinput_v2_head
        + offs_v * stride_dinput_v2_dim,
        end_v_prev2_grad,
        mask=v_mask,
    )


def compute_native_simamba_grads(
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Midpoint: Optional[torch.Tensor],
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles_Cumsum: torch.Tensor,
    D: Optional[torch.Tensor],
    grad_out: torch.Tensor,
    chunk_start_ssm_state: torch.Tensor,
    chunk_start_k_prev1_state: torch.Tensor,
    chunk_start_k_prev2_state: torch.Tensor,
    chunk_start_v_prev1_state: torch.Tensor,
    chunk_start_v_prev2_state: torch.Tensor,
    final_ssm_state: torch.Tensor,
    grad_final_ssm_state: Optional[torch.Tensor],
    grad_final_k_prev1_state: Optional[torch.Tensor],
    grad_final_k_prev2_state: Optional[torch.Tensor],
    grad_final_v_prev1_state: Optional[torch.Tensor],
    grad_final_v_prev2_state: Optional[torch.Tensor],
    chunk_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    ADT = ADT.contiguous()
    DT = DT.contiguous()
    Simpson = Simpson.contiguous()
    Q_bias = Q_bias.contiguous()
    K_bias = K_bias.contiguous()
    Angles_Cumsum = Angles_Cumsum.contiguous()
    grad_out = grad_out.contiguous()
    chunk_start_ssm_state = chunk_start_ssm_state.contiguous()
    chunk_start_k_prev1_state = chunk_start_k_prev1_state.contiguous()
    chunk_start_k_prev2_state = chunk_start_k_prev2_state.contiguous()
    chunk_start_v_prev1_state = chunk_start_v_prev1_state.contiguous()
    chunk_start_v_prev2_state = chunk_start_v_prev2_state.contiguous()
    final_ssm_state = final_ssm_state.contiguous()
    if Midpoint is not None:
        Midpoint = Midpoint.contiguous()
    if D is not None:
        D = D.contiguous()

    zero_k = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    zero_v = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    zero_ssm = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)

    grad_final_ssm_state = zero_ssm if grad_final_ssm_state is None else grad_final_ssm_state.contiguous()
    grad_final_k_prev1_state = zero_k if grad_final_k_prev1_state is None else grad_final_k_prev1_state.contiguous()
    grad_final_k_prev2_state = zero_k if grad_final_k_prev2_state is None else grad_final_k_prev2_state.contiguous()
    grad_final_v_prev1_state = zero_v if grad_final_v_prev1_state is None else grad_final_v_prev1_state.contiguous()
    grad_final_v_prev2_state = zero_v if grad_final_v_prev2_state is None else grad_final_v_prev2_state.contiguous()

    dq_partial = torch.empty((batch, seqlen, nheads, headdim_qk), device=Q.device, dtype=Q.dtype)
    dk_partial = torch.empty((batch, seqlen, nheads, headdim_qk), device=Q.device, dtype=K.dtype)
    dv = torch.empty_like(V)
    dadt = torch.empty_like(ADT, dtype=torch.float32)
    ddt = torch.empty_like(DT, dtype=torch.float32)
    dsimpson = torch.empty_like(Simpson, dtype=torch.float32)
    dmidpoint = torch.empty_like(Midpoint, dtype=torch.float32) if Midpoint is not None else None
    dangles_cumsum = torch.empty_like(Angles_Cumsum, dtype=torch.float32)
    dq_bias_partial = torch.empty((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    dk_bias_partial = torch.empty((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    dd_partial = torch.empty((batch, nheads), device=Q.device, dtype=torch.float32) if D is not None else None
    dinput_ssm = torch.empty((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    dinput_k_prev1 = torch.empty((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    dinput_k_prev2 = torch.empty((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    dinput_v_prev1 = torch.empty((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    dinput_v_prev2 = torch.empty((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)

    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    HEADDIM_V = triton.next_power_of_2(headdim_v)
    midpoint_ptr = Midpoint if Midpoint is not None else Simpson
    dmidpoint_ptr = dmidpoint if dmidpoint is not None else dsimpson
    dd_partial_ptr = dd_partial if dd_partial is not None else dq_bias_partial

    grid = (nheads, batch)
    _simamba_main_bwd_kernel[grid](
        Q,
        K,
        V,
        ADT,
        DT,
        Simpson,
        midpoint_ptr,
        Q_bias,
        K_bias,
        Angles_Cumsum,
        D,
        grad_out,
        chunk_start_ssm_state,
        chunk_start_k_prev1_state,
        chunk_start_k_prev2_state,
        chunk_start_v_prev1_state,
        chunk_start_v_prev2_state,
        final_ssm_state,
        grad_final_ssm_state,
        grad_final_k_prev1_state,
        grad_final_k_prev2_state,
        grad_final_v_prev1_state,
        grad_final_v_prev2_state,
        dq_partial,
        dk_partial,
        dv,
        dadt,
        ddt,
        dsimpson,
        dmidpoint_ptr,
        dangles_cumsum,
        dq_bias_partial,
        dk_bias_partial,
        dd_partial_ptr,
        dinput_ssm,
        dinput_k_prev1,
        dinput_k_prev2,
        dinput_v_prev1,
        dinput_v_prev2,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        ADT.stride(0), ADT.stride(1), ADT.stride(2),
        DT.stride(0), DT.stride(1), DT.stride(2),
        Simpson.stride(0), Simpson.stride(1), Simpson.stride(2),
        Midpoint.stride(0) if Midpoint is not None else 0,
        Midpoint.stride(1) if Midpoint is not None else 0,
        Midpoint.stride(2) if Midpoint is not None else 0,
        Q_bias.stride(0), Q_bias.stride(1),
        K_bias.stride(0), K_bias.stride(1),
        Angles_Cumsum.stride(0), Angles_Cumsum.stride(1), Angles_Cumsum.stride(2), Angles_Cumsum.stride(3),
        D.stride(0) if D is not None else 0,
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        chunk_start_ssm_state.stride(0), chunk_start_ssm_state.stride(1), chunk_start_ssm_state.stride(2), chunk_start_ssm_state.stride(3), chunk_start_ssm_state.stride(4),
        chunk_start_k_prev1_state.stride(0), chunk_start_k_prev1_state.stride(1), chunk_start_k_prev1_state.stride(2), chunk_start_k_prev1_state.stride(3),
        chunk_start_k_prev2_state.stride(0), chunk_start_k_prev2_state.stride(1), chunk_start_k_prev2_state.stride(2), chunk_start_k_prev2_state.stride(3),
        chunk_start_v_prev1_state.stride(0), chunk_start_v_prev1_state.stride(1), chunk_start_v_prev1_state.stride(2), chunk_start_v_prev1_state.stride(3),
        chunk_start_v_prev2_state.stride(0), chunk_start_v_prev2_state.stride(1), chunk_start_v_prev2_state.stride(2), chunk_start_v_prev2_state.stride(3),
        final_ssm_state.stride(0), final_ssm_state.stride(1), final_ssm_state.stride(2), final_ssm_state.stride(3),
        grad_final_ssm_state.stride(0), grad_final_ssm_state.stride(1), grad_final_ssm_state.stride(2), grad_final_ssm_state.stride(3),
        grad_final_k_prev1_state.stride(0), grad_final_k_prev1_state.stride(1), grad_final_k_prev1_state.stride(2),
        grad_final_k_prev2_state.stride(0), grad_final_k_prev2_state.stride(1), grad_final_k_prev2_state.stride(2),
        grad_final_v_prev1_state.stride(0), grad_final_v_prev1_state.stride(1), grad_final_v_prev1_state.stride(2),
        grad_final_v_prev2_state.stride(0), grad_final_v_prev2_state.stride(1), grad_final_v_prev2_state.stride(2),
        dq_partial.stride(0), dq_partial.stride(1), dq_partial.stride(2), dq_partial.stride(3),
        dk_partial.stride(0), dk_partial.stride(1), dk_partial.stride(2), dk_partial.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        dadt.stride(0), dadt.stride(1), dadt.stride(2),
        ddt.stride(0), ddt.stride(1), ddt.stride(2),
        dsimpson.stride(0), dsimpson.stride(1), dsimpson.stride(2),
        dmidpoint_ptr.stride(0), dmidpoint_ptr.stride(1), dmidpoint_ptr.stride(2),
        dangles_cumsum.stride(0), dangles_cumsum.stride(1), dangles_cumsum.stride(2), dangles_cumsum.stride(3),
        dq_bias_partial.stride(0), dq_bias_partial.stride(1), dq_bias_partial.stride(2),
        dk_bias_partial.stride(0), dk_bias_partial.stride(1), dk_bias_partial.stride(2),
        dd_partial_ptr.stride(0), dd_partial_ptr.stride(1),
        dinput_ssm.stride(0), dinput_ssm.stride(1), dinput_ssm.stride(2), dinput_ssm.stride(3),
        dinput_k_prev1.stride(0), dinput_k_prev1.stride(1), dinput_k_prev1.stride(2),
        dinput_k_prev2.stride(0), dinput_k_prev2.stride(1), dinput_k_prev2.stride(2),
        dinput_v_prev1.stride(0), dinput_v_prev1.stride(1), dinput_v_prev1.stride(2),
        dinput_v_prev2.stride(0), dinput_v_prev2.stride(1), dinput_v_prev2.stride(2),
        seqlen,
        nheads_qk,
        headdim_qk,
        headdim_v,
        Angles_Cumsum.shape[-1],
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        HAS_D=D is not None,
        HAS_MIDPOINT=Midpoint is not None,
    )

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        dq = dq_partial.reshape(batch, seqlen, nheads_qk, gqa_ratio, headdim_qk).sum(dim=3)
        dk = dk_partial.reshape(batch, seqlen, nheads_qk, gqa_ratio, headdim_qk).sum(dim=3)
    else:
        dq = dq_partial
        dk = dk_partial

    dq_bias = dq_bias_partial.sum(dim=0)
    dk_bias = dk_bias_partial.sum(dim=0)
    dD = dd_partial.sum(dim=0) if dd_partial is not None else None

    return (
        dq,
        dk,
        dv,
        dadt,
        ddt,
        dsimpson,
        dmidpoint,
        dangles_cumsum,
        dq_bias,
        dk_bias,
        dD,
        dinput_ssm,
        dinput_k_prev1,
        dinput_k_prev2,
        dinput_v_prev1,
        dinput_v_prev2,
    )


def _zero_input_states(
    *,
    batch: int,
    nheads: int,
    n_angles: int,
    headdim_qk: int,
    headdim_v: int,
    device: torch.device,
    q_dtype: torch.dtype,
    v_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((batch, nheads, n_angles), device=device, dtype=torch.float32),
        torch.zeros((batch, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32),
        torch.zeros((batch, nheads, headdim_qk), device=device, dtype=q_dtype),
        torch.zeros((batch, nheads, headdim_qk), device=device, dtype=q_dtype),
        torch.zeros((batch, nheads, headdim_v), device=device, dtype=v_dtype),
        torch.zeros((batch, nheads, headdim_v), device=device, dtype=v_dtype),
    )


def _compute_chunk_start_states(
    *,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Midpoint: Optional[torch.Tensor],
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    Input_States: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    chunk_size: int,
) -> Tuple[
    list[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    torch.Tensor,
]:
    batch, seqlen, nheads_qk, headdim_qk = K.shape
    _, _, nheads, headdim_v = V.shape
    n_angles = Angles.shape[-1]

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        K = K.repeat_interleave(gqa_ratio, dim=2)

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
        (
            input_angle_state,
            ssm_state,
            k_prev1,
            k_prev2,
            v_prev1,
            v_prev2,
        ) = _zero_input_states(
            batch=batch,
            nheads=nheads,
            n_angles=n_angles,
            headdim_qk=headdim_qk,
            headdim_v=headdim_v,
            device=K.device,
            q_dtype=K.dtype,
            v_dtype=V.dtype,
        )

    with torch.no_grad():
        angles_cumsum, _ = _compute_angle_cumsum(Angles, DT, input_angle_state)
        k_pre = K + K_bias[None, None, :, :]

        state_batch = []
        angle_running = input_angle_state.float()
        ssm_running = ssm_state.float()
        k_prev1_running = k_prev1.float()
        k_prev2_running = k_prev2.float()
        v_prev1_running = v_prev1.float()
        v_prev2_running = v_prev2.float()

        for start in range(0, seqlen, chunk_size):
            state_batch.append(
                (
                    angle_running.detach().clone(),
                    ssm_running.detach().clone(),
                    k_prev1_running.to(K.dtype).detach().clone(),
                    k_prev2_running.to(K.dtype).detach().clone(),
                    v_prev1_running.to(V.dtype).detach().clone(),
                    v_prev2_running.to(V.dtype).detach().clone(),
                )
            )
            end = min(seqlen, start + chunk_size)
            for t in range(start, end):
                k_t = _apply_pairwise_rotary(k_pre[:, t], angles_cumsum[:, t]).float()
                v_t = V[:, t].float()

                kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
                kv_prev1 = v_prev1_running.unsqueeze(-1) * k_prev1_running.unsqueeze(-2)
                kv_prev2 = v_prev2_running.unsqueeze(-1) * k_prev2_running.unsqueeze(-2)
                midpoint_t = Midpoint[:, :, t] if Midpoint is not None else None

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

                angle_running = angles_cumsum[:, t].float()
                k_prev2_running = k_prev1_running
                k_prev1_running = k_t
                v_prev2_running = v_prev1_running
                v_prev1_running = v_t

    return state_batch, angles_cumsum


def _chunked_reference_dt_grad(
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Midpoint: Optional[torch.Tensor],
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor],
    Z: Optional[torch.Tensor],
    grad_out: torch.Tensor,
    chunk_start_states: list[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    chunk_size: int,
) -> torch.Tensor:
    batch, seqlen, _, _ = Q.shape
    _, _, nheads, headdim_v = V.shape

    ddt = torch.empty_like(DT, dtype=torch.float32)
    next_state_grads = None

    for chunk_idx in range(len(chunk_start_states) - 1, -1, -1):
        start = chunk_idx * chunk_size
        end = min(seqlen, start + chunk_size)

        (
            angle_state,
            ssm_state,
            k_prev1,
            k_prev2,
            v_prev1,
            v_prev2,
        ) = chunk_start_states[chunk_idx]
        input_states_ref = (
            angle_state.detach().requires_grad_(True),
            ssm_state.detach().requires_grad_(True),
            k_prev1.detach().requires_grad_(True),
            k_prev2.detach().requires_grad_(True),
            v_prev1.detach().requires_grad_(True),
            v_prev2.detach().requires_grad_(True),
        )
        dt_ref = DT[:, :, start:end].detach().requires_grad_(True)

        with torch.enable_grad():
            outputs_ref = simamba_siso_combined(
                Q=Q[:, start:end].detach(),
                K=K[:, start:end].detach(),
                V=V[:, start:end].detach(),
                ADT=ADT[:, :, start:end].detach(),
                DT=dt_ref,
                Simpson=Simpson[:, :, start:end].detach(),
                Midpoint=Midpoint[:, :, start:end].detach() if Midpoint is not None else None,
                Q_bias=Q_bias.detach(),
                K_bias=K_bias.detach(),
                Angles=Angles[:, start:end].detach(),
                D=D.detach() if D is not None else None,
                Z=Z[:, start:end].detach() if Z is not None else None,
                Input_States=input_states_ref,
                return_final_states=True,
            )

        (
            out_ref,
            final_angle_ref,
            final_ssm_ref,
            final_k_prev1_ref,
            final_k_prev2_ref,
            final_v_prev1_ref,
            final_v_prev2_ref,
        ) = outputs_ref

        if next_state_grads is None:
            next_state_grads = (
                torch.zeros_like(final_angle_ref),
                torch.zeros_like(final_ssm_ref),
                torch.zeros_like(final_k_prev1_ref),
                torch.zeros_like(final_k_prev2_ref),
                torch.zeros_like(final_v_prev1_ref),
                torch.zeros_like(final_v_prev2_ref),
            )

        grad_pairs = [
            (out_ref, grad_out[:, start:end]),
            (final_angle_ref, next_state_grads[0]),
            (final_ssm_ref, next_state_grads[1]),
            (final_k_prev1_ref, next_state_grads[2]),
            (final_k_prev2_ref, next_state_grads[3]),
            (final_v_prev1_ref, next_state_grads[4]),
            (final_v_prev2_ref, next_state_grads[5]),
        ]
        active_outputs = [output for output, _ in grad_pairs if output.requires_grad]
        active_grad_outputs = [grad for output, grad in grad_pairs if output.requires_grad]
        grads = torch.autograd.grad(
            outputs=active_outputs,
            inputs=(dt_ref, *input_states_ref),
            grad_outputs=active_grad_outputs,
            allow_unused=True,
        )
        ddt[:, :, start:end] = grads[0] if grads[0] is not None else torch.zeros_like(dt_ref)
        next_state_grads = tuple(
            grad if grad is not None else torch.zeros_like(state)
            for grad, state in zip(grads[1:], input_states_ref)
        )

    return ddt


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
    recompute_chunk_size: int = 64,
    compute_dt_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute Simamba coefficient gradients.

    Notes:
    - dADT/dSimpson/dMidpoint come from an explicit streamed Simpson recurrence.
    - dDT is recovered with chunked reference recompute so it includes the
      DT->rotary-angle path without building a full-sequence autograd graph.
    """

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape

    if grad_out.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(
            f"grad_out shape mismatch: expected {(batch, seqlen, nheads, headdim_v)}, got {grad_out.shape}."
        )

    Q = Q.detach()
    K = K.detach()
    V = V.detach()
    ADT = ADT.detach()
    DT = DT.detach()
    Simpson = Simpson.detach()
    Midpoint = Midpoint.detach() if Midpoint is not None else None
    Q_bias = Q_bias.detach()
    K_bias = K_bias.detach()
    Angles = Angles.detach()
    D = D.detach() if D is not None else None
    Z = Z.detach() if Z is not None else None
    if Input_States is not None:
        Input_States = tuple(state.detach() for state in Input_States)

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa_ratio, dim=2)
        K = K.repeat_interleave(gqa_ratio, dim=2)

    chunk_size = max(1, min(recompute_chunk_size, seqlen))
    chunk_start_states, angles_cumsum = _compute_chunk_start_states(
        K=K,
        V=V,
        ADT=ADT,
        DT=DT,
        Simpson=Simpson,
        Midpoint=Midpoint,
        K_bias=K_bias,
        Angles=Angles,
        Input_States=Input_States,
        chunk_size=chunk_size,
    )

    q_pre = Q + Q_bias[None, None, :, :]
    k_pre = K + K_bias[None, None, :, :]

    if Z is not None:
        grad_eff = grad_out.float() * F.silu(Z.float())
    else:
        grad_eff = grad_out.float()

    dadt = torch.empty_like(ADT, dtype=torch.float32)
    ddt = torch.zeros_like(DT, dtype=torch.float32)
    dseff = torch.empty_like(Simpson, dtype=torch.float32)

    dssm_future = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)

    for chunk_idx in range(len(chunk_start_states) - 1, -1, -1):
        start = chunk_idx * chunk_size
        end = min(seqlen, start + chunk_size)
        chunk_len = end - start

        (
            _,
            ssm_start,
            k_prev1_start,
            k_prev2_start,
            v_prev1_start,
            v_prev2_start,
        ) = chunk_start_states[chunk_idx]

        ssm_prev_seq = torch.empty(
            (batch, chunk_len, nheads, headdim_v, headdim_qk),
            device=Q.device,
            dtype=torch.float32,
        )
        q_seq = torch.empty((batch, chunk_len, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        k_seq = torch.empty_like(q_seq)
        v_seq = torch.empty((batch, chunk_len, nheads, headdim_v), device=Q.device, dtype=torch.float32)
        s_eff = torch.empty((batch, nheads, chunk_len), device=Q.device, dtype=torch.float32)

        ssm_running = ssm_start.float()
        k_prev1_running = k_prev1_start.float()
        k_prev2_running = k_prev2_start.float()
        v_prev1_running = v_prev1_start.float()
        v_prev2_running = v_prev2_start.float()

        for local_t, global_t in enumerate(range(start, end)):
            ssm_prev_seq[:, local_t] = ssm_running

            q_t = _apply_pairwise_rotary(q_pre[:, global_t], angles_cumsum[:, global_t]).float()
            k_t = _apply_pairwise_rotary(k_pre[:, global_t], angles_cumsum[:, global_t]).float()
            v_t = V[:, global_t].float()

            q_seq[:, local_t] = q_t
            k_seq[:, local_t] = k_t
            v_seq[:, local_t] = v_t

            midpoint_t = Midpoint[:, :, global_t] if Midpoint is not None else None
            s_eff[:, :, local_t] = _resolve_simpson_effective(Simpson[:, :, global_t], midpoint_t)

            kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            kv_prev1 = v_prev1_running.unsqueeze(-1) * k_prev1_running.unsqueeze(-2)
            kv_prev2 = v_prev2_running.unsqueeze(-1) * k_prev2_running.unsqueeze(-2)

            ssm_running = _simpson_state_update(
                ssm_running,
                kv_t,
                kv_prev1,
                kv_prev2,
                ADT[:, :, global_t],
                DT[:, :, global_t],
                Simpson[:, :, global_t],
                midpoint_t,
            )

            k_prev2_running = k_prev1_running
            k_prev1_running = k_t
            v_prev2_running = v_prev1_running
            v_prev1_running = v_t

        alpha_chunk = torch.exp(ADT[:, :, start:end].float())
        boundary_k_prev1 = k_prev1_start.float()
        boundary_k_prev2 = k_prev2_start.float()
        boundary_v_prev1 = v_prev1_start.float()
        boundary_v_prev2 = v_prev2_start.float()

        for local_t in range(chunk_len - 1, -1, -1):
            global_t = start + local_t
            dssm_t = grad_eff[:, global_t].unsqueeze(-1) * q_seq[:, local_t].unsqueeze(-2) + dssm_future

            kv_t = v_seq[:, local_t].unsqueeze(-1) * k_seq[:, local_t].unsqueeze(-2)

            if local_t > 0:
                kv_prev1 = v_seq[:, local_t - 1].unsqueeze(-1) * k_seq[:, local_t - 1].unsqueeze(-2)
            else:
                kv_prev1 = boundary_v_prev1.unsqueeze(-1) * boundary_k_prev1.unsqueeze(-2)

            if local_t > 1:
                kv_prev2 = v_seq[:, local_t - 2].unsqueeze(-1) * k_seq[:, local_t - 2].unsqueeze(-2)
            elif local_t == 1:
                kv_prev2 = boundary_v_prev1.unsqueeze(-1) * boundary_k_prev1.unsqueeze(-2)
            else:
                kv_prev2 = boundary_v_prev2.unsqueeze(-1) * boundary_k_prev2.unsqueeze(-2)

            dadt_t, ddt_t, dseff_t = _compute_step_coeff_grads_triton(
                dssm=dssm_t,
                ssm_prev=ssm_prev_seq[:, local_t],
                kv_t=kv_t,
                kv_prev1=kv_prev1,
                kv_prev2=kv_prev2,
                adt_t=ADT[:, :, global_t].float(),
                dt_t=DT[:, :, global_t].float(),
                s_eff_t=s_eff[:, :, local_t],
            )

            dadt[:, :, global_t] = dadt_t
            if compute_dt_grad:
                ddt[:, :, global_t] = ddt_t
            dseff[:, :, global_t] = dseff_t
            dssm_future = alpha_chunk[:, :, local_t].unsqueeze(-1).unsqueeze(-1) * dssm_t

    if Midpoint is None:
        dsimpson = dseff
        dmidpoint = None
    else:
        dsimpson = dseff * Midpoint.float()
        dmidpoint = dseff * Simpson.float()

    if compute_dt_grad:
        ddt = _chunked_reference_dt_grad(
            Q=Q,
            K=K,
            V=V,
            ADT=ADT,
            DT=DT,
            Simpson=Simpson,
            Midpoint=Midpoint,
            Q_bias=Q_bias,
            K_bias=K_bias,
            Angles=Angles,
            D=D,
            Z=Z,
            grad_out=grad_out.float(),
            chunk_start_states=chunk_start_states,
            chunk_size=chunk_size,
        )

    return dadt, ddt, dsimpson, dmidpoint
