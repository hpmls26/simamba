"""Simamba SISO forward path.

This file has two inference implementations:
- a chunk-parallel Triton prefill kernel for batched inference
- a fallback loop over the Triton step kernel for varlen or training-oriented
  code paths that still need wrapper simplicity
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx, silu
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step


def _alloc_state_buffers_like(states):
    return tuple(torch.empty_like(state) for state in states)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "HAS_D", "HAS_Z", "HAS_MIDPOINT", "HAS_INITIAL_STATES", "RETURN_FINAL_STATES"],
)
@triton.jit
def mamba3_siso_fwd_kernel(
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
    Z,
    Initial_SSM_State,
    Initial_K_prev1_State,
    Initial_K_prev2_State,
    Initial_V_prev1_State,
    Initial_V_prev2_State,
    Out,
    Out_Pregate,
    Final_SSM_State,
    Final_K_prev1_State,
    Final_K_prev2_State,
    Final_V_prev1_State,
    Final_V_prev2_State,
    Chunk_Start_SSM_State,
    Chunk_Start_K_prev1_State,
    Chunk_Start_K_prev2_State,
    Chunk_Start_V_prev1_State,
    Chunk_Start_V_prev2_State,
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
    stride_z_batch,
    stride_z_seqlen,
    stride_z_head,
    stride_z_vdim,
    stride_init_ssm_batch,
    stride_init_ssm_head,
    stride_init_ssm_vdim,
    stride_init_ssm_qkdim,
    stride_init_k_prev1_batch,
    stride_init_k_prev1_head,
    stride_init_k_prev1_dim,
    stride_init_k_prev2_batch,
    stride_init_k_prev2_head,
    stride_init_k_prev2_dim,
    stride_init_v_prev1_batch,
    stride_init_v_prev1_head,
    stride_init_v_prev1_dim,
    stride_init_v_prev2_batch,
    stride_init_v_prev2_head,
    stride_init_v_prev2_dim,
    stride_o_batch,
    stride_o_seqlen,
    stride_o_head,
    stride_o_vdim,
    stride_o_pregate_batch,
    stride_o_pregate_seqlen,
    stride_o_pregate_head,
    stride_o_pregate_vdim,
    stride_final_ssm_batch,
    stride_final_ssm_head,
    stride_final_ssm_vdim,
    stride_final_ssm_qkdim,
    stride_final_k_prev1_batch,
    stride_final_k_prev1_head,
    stride_final_k_prev1_dim,
    stride_final_k_prev2_batch,
    stride_final_k_prev2_head,
    stride_final_k_prev2_dim,
    stride_final_v_prev1_batch,
    stride_final_v_prev1_head,
    stride_final_v_prev1_dim,
    stride_final_v_prev2_batch,
    stride_final_v_prev2_head,
    stride_final_v_prev2_dim,
    stride_chunk_ssm_batch,
    stride_chunk_ssm_head,
    stride_chunk_ssm_chunk,
    stride_chunk_ssm_vdim,
    stride_chunk_ssm_qkdim,
    stride_chunk_k_prev1_batch,
    stride_chunk_k_prev1_head,
    stride_chunk_k_prev1_chunk,
    stride_chunk_k_prev1_dim,
    stride_chunk_k_prev2_batch,
    stride_chunk_k_prev2_head,
    stride_chunk_k_prev2_chunk,
    stride_chunk_k_prev2_dim,
    stride_chunk_v_prev1_batch,
    stride_chunk_v_prev1_head,
    stride_chunk_v_prev1_chunk,
    stride_chunk_v_prev1_dim,
    stride_chunk_v_prev2_batch,
    stride_chunk_v_prev2_head,
    stride_chunk_v_prev2_chunk,
    stride_chunk_v_prev2_dim,
    seqlen,
    nheads_qk,
    headdim_angles,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_MIDPOINT: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    RETURN_FINAL_STATES: tl.constexpr,
    STORE_CHUNK_BOUNDARY_STATES: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    offs_s = tl.arange(0, CHUNK_SIZE)
    offs_qk = tl.arange(0, HEADDIM_QK)
    offs_v = tl.arange(0, HEADDIM_V)
    offs_angle = tl.arange(0, HEADDIM_QK // 2)
    log2e = 1.44269504089
    half_log2e = 0.721347520445

    q_bias = tl.load(Q_bias + pid_head * stride_q_bias_head + offs_qk * stride_q_bias_qkdim)
    k_bias = tl.load(K_bias + pid_head * stride_k_bias_head + offs_qk * stride_k_bias_qkdim)

    if HAS_D:
        D_val = tl.load(D + pid_head * stride_d_head).to(tl.float32)
    else:
        D_val = 0.0

    if HAS_INITIAL_STATES:
        acc_ssm = tl.load(
            Initial_SSM_State
            + pid_batch * stride_init_ssm_batch
            + pid_head * stride_init_ssm_head
            + offs_v[:, None] * stride_init_ssm_vdim
            + offs_qk[None, :] * stride_init_ssm_qkdim,
        ).to(tl.float32)
        k_prev1 = tl.load(
            Initial_K_prev1_State
            + pid_batch * stride_init_k_prev1_batch
            + pid_head * stride_init_k_prev1_head
            + offs_qk * stride_init_k_prev1_dim,
        ).to(tl.float32)
        k_prev2 = tl.load(
            Initial_K_prev2_State
            + pid_batch * stride_init_k_prev2_batch
            + pid_head * stride_init_k_prev2_head
            + offs_qk * stride_init_k_prev2_dim,
        ).to(tl.float32)
        v_prev1 = tl.load(
            Initial_V_prev1_State
            + pid_batch * stride_init_v_prev1_batch
            + pid_head * stride_init_v_prev1_head
            + offs_v * stride_init_v_prev1_dim,
        ).to(tl.float32)
        v_prev2 = tl.load(
            Initial_V_prev2_State
            + pid_batch * stride_init_v_prev2_batch
            + pid_head * stride_init_v_prev2_head
            + offs_v * stride_init_v_prev2_dim,
        ).to(tl.float32)
    else:
        acc_ssm = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)
        k_prev1 = tl.zeros([HEADDIM_QK], dtype=tl.float32)
        k_prev2 = tl.zeros([HEADDIM_QK], dtype=tl.float32)
        v_prev1 = tl.zeros([HEADDIM_V], dtype=tl.float32)
        v_prev2 = tl.zeros([HEADDIM_V], dtype=tl.float32)

    for chunk_start in range(0, seqlen, CHUNK_SIZE):
        chunk_idx = chunk_start // CHUNK_SIZE
        offs_seq = chunk_start + offs_s
        seq_mask = offs_seq < seqlen

        if STORE_CHUNK_BOUNDARY_STATES:
            tl.store(
                Chunk_Start_SSM_State
                + pid_batch * stride_chunk_ssm_batch
                + pid_head * stride_chunk_ssm_head
                + chunk_idx * stride_chunk_ssm_chunk
                + offs_v[:, None] * stride_chunk_ssm_vdim
                + offs_qk[None, :] * stride_chunk_ssm_qkdim,
                acc_ssm,
            )
            tl.store(
                Chunk_Start_K_prev1_State
                + pid_batch * stride_chunk_k_prev1_batch
                + pid_head * stride_chunk_k_prev1_head
                + chunk_idx * stride_chunk_k_prev1_chunk
                + offs_qk * stride_chunk_k_prev1_dim,
                k_prev1,
            )
            tl.store(
                Chunk_Start_K_prev2_State
                + pid_batch * stride_chunk_k_prev2_batch
                + pid_head * stride_chunk_k_prev2_head
                + chunk_idx * stride_chunk_k_prev2_chunk
                + offs_qk * stride_chunk_k_prev2_dim,
                k_prev2,
            )
            tl.store(
                Chunk_Start_V_prev1_State
                + pid_batch * stride_chunk_v_prev1_batch
                + pid_head * stride_chunk_v_prev1_head
                + chunk_idx * stride_chunk_v_prev1_chunk
                + offs_v * stride_chunk_v_prev1_dim,
                v_prev1,
            )
            tl.store(
                Chunk_Start_V_prev2_State
                + pid_batch * stride_chunk_v_prev2_batch
                + pid_head * stride_chunk_v_prev2_head
                + chunk_idx * stride_chunk_v_prev2_chunk
                + offs_v * stride_chunk_v_prev2_dim,
                v_prev2,
            )

        q_pre = tl.load(
            Q + pid_batch * stride_q_batch + offs_seq[:, None] * stride_q_seqlen + head_idx_qk * stride_q_head + offs_qk[None, :] * stride_q_qkdim,
            mask=seq_mask[:, None],
            other=0.0,
        )
        k_pre = tl.load(
            K + pid_batch * stride_k_batch + offs_seq[:, None] * stride_k_seqlen + head_idx_qk * stride_k_head + offs_qk[None, :] * stride_k_qkdim,
            mask=seq_mask[:, None],
            other=0.0,
        )
        v_block = tl.load(
            V + pid_batch * stride_v_batch + offs_seq[:, None] * stride_v_seqlen + pid_head * stride_v_head + offs_v[None, :] * stride_v_vdim,
            mask=seq_mask[:, None],
            other=0.0,
        )

        da = tl.load(
            ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head + offs_seq * stride_adt_seqlen,
            mask=seq_mask,
            other=0.0,
        ).to(tl.float32) * log2e
        dt = tl.load(
            DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head + offs_seq * stride_dt_seqlen,
            mask=seq_mask,
            other=0.0,
        ).to(tl.float32)
        simpson = tl.load(
            Simpson + pid_batch * stride_simpson_batch + pid_head * stride_simpson_head + offs_seq * stride_simpson_seqlen,
            mask=seq_mask,
            other=0.0,
        ).to(tl.float32)
        simpson = tl.minimum(1.0, tl.maximum(0.0, simpson))
        if HAS_MIDPOINT:
            midpoint = tl.load(
                Midpoint + pid_batch * stride_midpoint_batch + pid_head * stride_midpoint_head + offs_seq * stride_midpoint_seqlen,
                mask=seq_mask,
                other=0.0,
            ).to(tl.float32)
            midpoint = tl.minimum(1.0, tl.maximum(0.0, midpoint))
            simpson = tl.minimum(1.0, tl.maximum(0.0, simpson * midpoint))

        angle_block = tl.load(
            Angles_Cumsum + pid_batch * stride_angles_batch + offs_seq[:, None] * stride_angles_seqlen + pid_head * stride_angles_head + offs_angle[None, :] * stride_angles_qkdim,
            mask=seq_mask[:, None] & (offs_angle[None, :] < headdim_angles),
            other=0.0,
        ).to(tl.float32)

        if HAS_Z:
            z_block = tl.load(
                Z + pid_batch * stride_z_batch + offs_seq[:, None] * stride_z_seqlen + pid_head * stride_z_head + offs_v[None, :] * stride_z_vdim,
                mask=seq_mask[:, None],
                other=0.0,
            )

        q_pre += q_bias[None, :]
        k_pre += k_bias[None, :]

        cos_block = cos_approx(angle_block)
        sin_block = sin_approx(angle_block)

        q0, q1 = tl.split(tl.reshape(q_pre, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
        qo0 = q0 * cos_block - q1 * sin_block
        qo1 = q0 * sin_block + q1 * cos_block
        q_rot = tl.reshape(tl.join(qo0, qo1), [CHUNK_SIZE, HEADDIM_QK]).to(q_pre.dtype)

        k0, k1 = tl.split(tl.reshape(k_pre, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
        ko0 = k0 * cos_block - k1 * sin_block
        ko1 = k0 * sin_block + k1 * cos_block
        k_rot = tl.reshape(tl.join(ko0, ko1), [CHUNK_SIZE, HEADDIM_QK]).to(k_pre.dtype)

        alpha = tl.math.exp2(da)
        alpha_half = tl.math.exp2(da * 0.5)
        gamma0 = (dt / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * simpson))
        gamma1 = (dt / 6.0) * (alpha + alpha_half * (2.0 + simpson))
        gamma2 = -(dt / 12.0) * alpha_half * simpson
        gamma0 = tl.where(seq_mask, gamma0, 0.0)
        gamma1 = tl.where(seq_mask, gamma1, 0.0)
        gamma2 = tl.where(seq_mask, gamma2, 0.0)

        shift1_mask = seq_mask & (offs_s + 1 < CHUNK_SIZE) & (offs_seq + 1 < seqlen)
        shift2_mask = seq_mask & (offs_s + 2 < CHUNK_SIZE) & (offs_seq + 2 < seqlen)

        da_next1 = tl.load(
            ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head + (offs_seq + 1) * stride_adt_seqlen,
            mask=shift1_mask,
            other=0.0,
        ).to(tl.float32) * log2e
        dt_next1 = tl.load(
            DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head + (offs_seq + 1) * stride_dt_seqlen,
            mask=shift1_mask,
            other=0.0,
        ).to(tl.float32)
        simpson_next1 = tl.load(
            Simpson + pid_batch * stride_simpson_batch + pid_head * stride_simpson_head + (offs_seq + 1) * stride_simpson_seqlen,
            mask=shift1_mask,
            other=0.0,
        ).to(tl.float32)
        simpson_next1 = tl.minimum(1.0, tl.maximum(0.0, simpson_next1))
        if HAS_MIDPOINT:
            midpoint_next1 = tl.load(
                Midpoint + pid_batch * stride_midpoint_batch + pid_head * stride_midpoint_head + (offs_seq + 1) * stride_midpoint_seqlen,
                mask=shift1_mask,
                other=0.0,
            ).to(tl.float32)
            midpoint_next1 = tl.minimum(1.0, tl.maximum(0.0, midpoint_next1))
            simpson_next1 = tl.minimum(1.0, tl.maximum(0.0, simpson_next1 * midpoint_next1))
        alpha_next1 = tl.math.exp2(da_next1)
        alpha_half_next1 = tl.math.exp2(da_next1 * 0.5)
        gamma1_shift = (dt_next1 / 6.0) * (alpha_next1 + alpha_half_next1 * (2.0 + simpson_next1))
        gamma1_shift = tl.where(shift1_mask, gamma1_shift, 0.0)

        da_next2 = tl.load(
            ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head + (offs_seq + 2) * stride_adt_seqlen,
            mask=shift2_mask,
            other=0.0,
        ).to(tl.float32) * log2e
        dt_next2 = tl.load(
            DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head + (offs_seq + 2) * stride_dt_seqlen,
            mask=shift2_mask,
            other=0.0,
        ).to(tl.float32)
        simpson_next2 = tl.load(
            Simpson + pid_batch * stride_simpson_batch + pid_head * stride_simpson_head + (offs_seq + 2) * stride_simpson_seqlen,
            mask=shift2_mask,
            other=0.0,
        ).to(tl.float32)
        simpson_next2 = tl.minimum(1.0, tl.maximum(0.0, simpson_next2))
        if HAS_MIDPOINT:
            midpoint_next2 = tl.load(
                Midpoint + pid_batch * stride_midpoint_batch + pid_head * stride_midpoint_head + (offs_seq + 2) * stride_midpoint_seqlen,
                mask=shift2_mask,
                other=0.0,
            ).to(tl.float32)
            midpoint_next2 = tl.minimum(1.0, tl.maximum(0.0, midpoint_next2))
            simpson_next2 = tl.minimum(1.0, tl.maximum(0.0, simpson_next2 * midpoint_next2))
        alpha_half_next2 = tl.math.exp2(da_next2 * 0.5)
        gamma2_shift = -(dt_next2 / 12.0) * alpha_half_next2 * simpson_next2
        gamma2_shift = tl.where(shift2_mask, gamma2_shift, 0.0)

        da_cs = tl.cumsum(da)
        exp_da = tl.math.exp2(da_cs)
        inv_exp_da = tl.math.exp2(-da_cs)
        da_cs_last = tl.sum(da, axis=0)
        exp_da_last = tl.math.exp2(da_cs_last)

        scale0 = gamma0 * inv_exp_da
        scale1 = gamma1_shift * inv_exp_da * tl.math.exp2(-da_next1)
        scale2 = gamma2_shift * inv_exp_da * tl.math.exp2(-(da_next1 + da_next2))

        k0_scaled = k_rot * scale0[:, None]
        k1_scaled = k_rot * scale1[:, None]
        k2_scaled = k_rot * scale2[:, None]

        acc_o = tl.dot(q_rot, tl.trans(acc_ssm).to(q_rot.dtype))
        acc_o *= exp_da[:, None]

        row0 = offs_s == 0
        row1 = offs_s == 1
        da_cs0 = tl.sum(tl.where(row0, da_cs, 0.0), axis=0)
        da_cs1 = tl.sum(tl.where(row1, da_cs, 0.0), axis=0)
        gamma1_0 = tl.sum(tl.where(row0, gamma1, 0.0), axis=0)
        gamma2_0 = tl.sum(tl.where(row0, gamma2, 0.0), axis=0)
        gamma2_1 = tl.sum(tl.where(row1, gamma2, 0.0), axis=0)

        ext0 = tl.math.exp2(-da_cs0) * (
            gamma1_0 * (v_prev1[:, None] * k_prev1[None, :])
            + gamma2_0 * (v_prev2[:, None] * k_prev2[None, :])
        )
        acc_o += tl.dot(q_rot, tl.trans(ext0).to(q_rot.dtype)) * exp_da[:, None]

        ext1 = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)
        if CHUNK_SIZE > 1:
            ext1 = tl.math.exp2(-da_cs1) * gamma2_1 * (v_prev1[:, None] * k_prev1[None, :])
            acc_o += tl.dot(q_rot, tl.trans(ext1).to(q_rot.dtype)) * exp_da[:, None] * (offs_s[:, None] >= 1)

        s0 = tl.dot(q_rot, tl.trans(k0_scaled).to(q_rot.dtype))
        s0 *= exp_da[:, None]
        s0 = tl.where(offs_s[:, None] >= offs_s[None, :], s0, 0.0)
        acc_o += tl.dot(s0.to(v_block.dtype), v_block)

        s1 = tl.dot(q_rot, tl.trans(k1_scaled).to(q_rot.dtype))
        s1 *= exp_da[:, None]
        s1 = tl.where(offs_s[:, None] >= (offs_s[None, :] + 1), s1, 0.0)
        acc_o += tl.dot(s1.to(v_block.dtype), v_block)

        s2 = tl.dot(q_rot, tl.trans(k2_scaled).to(q_rot.dtype))
        s2 *= exp_da[:, None]
        s2 = tl.where(offs_s[:, None] >= (offs_s[None, :] + 2), s2, 0.0)
        acc_o += tl.dot(s2.to(v_block.dtype), v_block)

        if HAS_D:
            acc_o += D_val * v_block.to(tl.float32)
        if STORE_CHUNK_BOUNDARY_STATES:
            tl.store(
                Out_Pregate
                + pid_batch * stride_o_pregate_batch
                + offs_seq[:, None] * stride_o_pregate_seqlen
                + pid_head * stride_o_pregate_head
                + offs_v[None, :] * stride_o_pregate_vdim,
                acc_o,
                mask=seq_mask[:, None],
            )
        if HAS_Z:
            acc_o = acc_o * silu(z_block.to(tl.float32))

        tl.store(
            Out + pid_batch * stride_o_batch + offs_seq[:, None] * stride_o_seqlen + pid_head * stride_o_head + offs_v[None, :] * stride_o_vdim,
            acc_o,
            mask=seq_mask[:, None],
        )

        total_scale = exp_da_last * (scale0 + scale1 + scale2)
        acc_ssm = exp_da_last * acc_ssm + tl.dot(
            tl.trans(v_block).to(k_rot.dtype),
            (k_rot * total_scale[:, None]).to(k_rot.dtype),
        ) + exp_da_last * ext0 + exp_da_last * ext1

        valid_count = tl.minimum(CHUNK_SIZE, seqlen - chunk_start)
        last_row = valid_count - 1
        second_last_row = valid_count - 2
        last_mask = offs_s == last_row
        second_last_mask = offs_s == second_last_row
        prev_k_prev1 = k_prev1
        prev_v_prev1 = v_prev1
        k_last = tl.sum(k_rot.to(tl.float32) * last_mask[:, None], axis=0)
        v_last = tl.sum(v_block.to(tl.float32) * last_mask[:, None], axis=0)
        k_second = tl.sum(k_rot.to(tl.float32) * second_last_mask[:, None], axis=0)
        v_second = tl.sum(v_block.to(tl.float32) * second_last_mask[:, None], axis=0)
        k_prev2 = tl.where(valid_count > 1, k_second, prev_k_prev1)
        v_prev2 = tl.where(valid_count > 1, v_second, prev_v_prev1)
        k_prev1 = k_last
        v_prev1 = v_last

    if RETURN_FINAL_STATES:
        tl.store(
            Final_SSM_State
            + pid_batch * stride_final_ssm_batch
            + pid_head * stride_final_ssm_head
            + offs_v[:, None] * stride_final_ssm_vdim
            + offs_qk[None, :] * stride_final_ssm_qkdim,
            acc_ssm,
        )
        tl.store(
            Final_K_prev1_State + pid_batch * stride_final_k_prev1_batch + pid_head * stride_final_k_prev1_head + offs_qk * stride_final_k_prev1_dim,
            k_prev1,
        )
        tl.store(
            Final_K_prev2_State + pid_batch * stride_final_k_prev2_batch + pid_head * stride_final_k_prev2_head + offs_qk * stride_final_k_prev2_dim,
            k_prev2,
        )
        tl.store(
            Final_V_prev1_State + pid_batch * stride_final_v_prev1_batch + pid_head * stride_final_v_prev1_head + offs_v * stride_final_v_prev1_dim,
            v_prev1,
        )
        tl.store(
            Final_V_prev2_State + pid_batch * stride_final_v_prev2_batch + pid_head * stride_final_v_prev2_head + offs_v * stride_final_v_prev2_dim,
            v_prev2,
        )


def _mamba3_siso_fwd_loop(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    Midpoint: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = None,
    return_final_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    n_angles = Angles.shape[-1]

    is_varlen = cu_seqlens is not None
    if is_varlen:
        num_sequences = cu_seqlens.shape[0] - 1
        state_batch = num_sequences
    else:
        num_sequences = batch
        state_batch = batch

    if Initial_States is None:
        init_angle_state = torch.zeros((state_batch, nheads, n_angles), device=Q.device, dtype=torch.float32)
        init_ssm_state = torch.zeros((state_batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
        init_k_prev1_state = torch.zeros((state_batch, nheads, headdim_qk), device=Q.device, dtype=Q.dtype)
        init_k_prev2_state = torch.zeros((state_batch, nheads, headdim_qk), device=Q.device, dtype=Q.dtype)
        init_v_prev1_state = torch.zeros((state_batch, nheads, headdim_v), device=Q.device, dtype=V.dtype)
        init_v_prev2_state = torch.zeros((state_batch, nheads, headdim_v), device=Q.device, dtype=V.dtype)
    else:
        (
            init_angle_state,
            init_ssm_state,
            init_k_prev1_state,
            init_k_prev2_state,
            init_v_prev1_state,
            init_v_prev2_state,
        ) = Initial_States

    out = torch.empty((batch, seqlen, nheads, headdim_v), device=V.device, dtype=V.dtype)

    if not is_varlen:
        states = (
            init_angle_state,
            init_ssm_state,
            init_k_prev1_state,
            init_k_prev2_state,
            init_v_prev1_state,
            init_v_prev2_state,
        )
        next_states = _alloc_state_buffers_like(states)
        for t in range(seqlen):
            _, next_states = mamba3_siso_step(
                Q=Q[:, t],
                K=K[:, t],
                V=V[:, t],
                ADT=ADT[:, :, t],
                DT=DT[:, :, t],
                Simpson=Simpson[:, :, t],
                Q_bias=Q_bias,
                K_bias=K_bias,
                Angles=Angles[:, t],
                Midpoint=Midpoint[:, :, t] if Midpoint is not None else None,
                D=D,
                Z=Z[:, t] if Z is not None else None,
                Out=out[:, t],
                Input_States=states,
                Output_States=next_states,
            )
            states, next_states = next_states, states
        final_states = states if return_final_states else None
    else:
        final_angle_states = []
        final_ssm_states = []
        final_k_prev1_states = []
        final_k_prev2_states = []
        final_v_prev1_states = []
        final_v_prev2_states = []

        for seq_idx in range(num_sequences):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            states = (
                init_angle_state[seq_idx : seq_idx + 1],
                init_ssm_state[seq_idx : seq_idx + 1],
                init_k_prev1_state[seq_idx : seq_idx + 1],
                init_k_prev2_state[seq_idx : seq_idx + 1],
                init_v_prev1_state[seq_idx : seq_idx + 1],
                init_v_prev2_state[seq_idx : seq_idx + 1],
            )
            next_states = _alloc_state_buffers_like(states)
            for t in range(start, end):
                _, next_states = mamba3_siso_step(
                    Q=Q[:, t],
                    K=K[:, t],
                    V=V[:, t],
                    ADT=ADT[:, :, t],
                    DT=DT[:, :, t],
                    Simpson=Simpson[:, :, t],
                    Q_bias=Q_bias,
                    K_bias=K_bias,
                    Angles=Angles[:, t],
                    Midpoint=Midpoint[:, :, t] if Midpoint is not None else None,
                    D=D,
                    Z=Z[:, t] if Z is not None else None,
                    Out=out[:, t],
                    Input_States=states,
                    Output_States=next_states,
                )
                states, next_states = next_states, states
            if return_final_states:
                final_angle_states.append(states[0])
                final_ssm_states.append(states[1])
                final_k_prev1_states.append(states[2])
                final_k_prev2_states.append(states[3])
                final_v_prev1_states.append(states[4])
                final_v_prev2_states.append(states[5])

        final_states = None
        if return_final_states:
            final_states = (
                torch.cat(final_angle_states, dim=0),
                torch.cat(final_ssm_states, dim=0),
                torch.cat(final_k_prev1_states, dim=0),
                torch.cat(final_k_prev2_states, dim=0),
                torch.cat(final_v_prev1_states, dim=0),
                torch.cat(final_v_prev2_states, dim=0),
            )

    return out, final_states


def mamba3_siso_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Simpson: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    Midpoint: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = None,
    chunk_size: int = 64,
    store_states_adt_outv: bool = False,
    return_final_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    if cu_seqlens is not None:
        raise NotImplementedError("Simamba Triton forward does not support cu_seqlens / variable-length mode.")
    if K.shape != Q.shape:
        raise ValueError(f"Q and K shape mismatch: {Q.shape} vs {K.shape}.")
    _, _, nheads, headdim_v = V.shape

    if nheads % nheads_qk != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk}).")
    if ADT.shape != (batch, nheads, seqlen):
        raise ValueError(f"ADT shape mismatch: expected {(batch, nheads, seqlen)}, got {ADT.shape}.")
    if DT.shape != (batch, nheads, seqlen):
        raise ValueError(f"DT shape mismatch: expected {(batch, nheads, seqlen)}, got {DT.shape}.")
    if Simpson.shape != (batch, nheads, seqlen):
        raise ValueError(f"Simpson shape mismatch: expected {(batch, nheads, seqlen)}, got {Simpson.shape}.")
    if Midpoint is not None and Midpoint.shape != (batch, nheads, seqlen):
        raise ValueError(f"Midpoint shape mismatch: expected {(batch, nheads, seqlen)}, got {Midpoint.shape}.")
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"Q_bias shape mismatch: expected {(nheads, headdim_qk)}, got {Q_bias.shape}.")
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"K_bias shape mismatch: expected {(nheads, headdim_qk)}, got {K_bias.shape}.")
    if D is not None and D.shape != (nheads,):
        raise ValueError(f"D shape mismatch: expected {(nheads,)}, got {D.shape}.")
    if Z is not None and Z.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(f"Z shape mismatch: expected {(batch, seqlen, nheads, headdim_v)}, got {Z.shape}.")

    n_angles = Angles.shape[-1]
    if Angles.shape != (batch, seqlen, nheads, n_angles):
        raise ValueError(f"Angles shape mismatch: expected {(batch, seqlen, nheads, n_angles)}, got {Angles.shape}.")

    # The chunk-parallel kernel uses tl.dot over the qk axis, which on current
    # Triton requires K >= 16. Tiny test shapes fall back to the step loop.
    if headdim_qk < 16:
        out, final_states = _mamba3_siso_fwd_loop(
            Q=Q,
            K=K,
            V=V,
            ADT=ADT,
            DT=DT,
            Simpson=Simpson,
            Q_bias=Q_bias,
            K_bias=K_bias,
            Angles=Angles,
            Midpoint=Midpoint,
            D=D,
            Z=Z,
            Initial_States=Initial_States,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )
        out_pregate = None
        angles_cumsum = None
        chunk_ssm_starts = None
        chunk_k_prev1_starts = None
        chunk_k_prev2_starts = None
        chunk_v_prev1_starts = None
        chunk_v_prev2_starts = None
    else:
        state_batch = batch
        nchunks = triton.cdiv(seqlen, chunk_size)
        if Initial_States is None:
            init_angle_state = None
            init_ssm_state = torch.zeros((state_batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
            init_k_prev1_state = torch.zeros((state_batch, nheads, headdim_qk), device=Q.device, dtype=Q.dtype)
            init_k_prev2_state = torch.zeros((state_batch, nheads, headdim_qk), device=Q.device, dtype=Q.dtype)
            init_v_prev1_state = torch.zeros((state_batch, nheads, headdim_v), device=Q.device, dtype=V.dtype)
            init_v_prev2_state = torch.zeros((state_batch, nheads, headdim_v), device=Q.device, dtype=V.dtype)
        else:
            (
                init_angle_state,
                init_ssm_state,
                init_k_prev1_state,
                init_k_prev2_state,
                init_v_prev1_state,
                init_v_prev2_state,
            ) = Initial_States

        angles_cumsum, final_angle_state = angle_dt_fwd(
            Angles,
            DT,
            init_state=init_angle_state,
            chunk_size=chunk_size,
            return_output_state=True,
            cu_seqlens=None,
        )

        out = torch.empty((batch, seqlen, nheads, headdim_v), device=V.device, dtype=V.dtype)
        out_pregate = torch.empty_like(out) if store_states_adt_outv else None
        needs_final_state_buffers = return_final_states or store_states_adt_outv
        if needs_final_state_buffers:
            final_ssm_state = torch.empty_like(init_ssm_state)
            final_k_prev1_state = torch.empty_like(init_k_prev1_state)
            final_k_prev2_state = torch.empty_like(init_k_prev2_state)
            final_v_prev1_state = torch.empty_like(init_v_prev1_state)
            final_v_prev2_state = torch.empty_like(init_v_prev2_state)
        else:
            final_ssm_state = torch.empty((1,), device=Q.device, dtype=torch.float32)
            final_k_prev1_state = torch.empty((1,), device=Q.device, dtype=Q.dtype)
            final_k_prev2_state = torch.empty((1,), device=Q.device, dtype=Q.dtype)
            final_v_prev1_state = torch.empty((1,), device=Q.device, dtype=V.dtype)
            final_v_prev2_state = torch.empty((1,), device=Q.device, dtype=V.dtype)

        if store_states_adt_outv:
            chunk_ssm_starts = torch.empty(
                (batch, nheads, nchunks, headdim_v, headdim_qk),
                device=Q.device,
                dtype=torch.float32,
            )
            chunk_k_prev1_starts = torch.empty(
                (batch, nheads, nchunks, headdim_qk),
                device=Q.device,
                dtype=Q.dtype,
            )
            chunk_k_prev2_starts = torch.empty_like(chunk_k_prev1_starts)
            chunk_v_prev1_starts = torch.empty(
                (batch, nheads, nchunks, headdim_v),
                device=Q.device,
                dtype=V.dtype,
            )
            chunk_v_prev2_starts = torch.empty_like(chunk_v_prev1_starts)
        else:
            chunk_ssm_starts = None
            chunk_k_prev1_starts = None
            chunk_k_prev2_starts = None
            chunk_v_prev1_starts = None
            chunk_v_prev2_starts = None

        midpoint_ptr = Midpoint if Midpoint is not None else Simpson
        grid = (nheads, batch)
        mamba3_siso_fwd_kernel[grid](
            Q,
            K,
            V,
            ADT,
            DT,
            Simpson,
            midpoint_ptr,
            Q_bias,
            K_bias,
            angles_cumsum,
            D,
            Z,
            init_ssm_state,
            init_k_prev1_state,
            init_k_prev2_state,
            init_v_prev1_state,
            init_v_prev2_state,
            out,
            out_pregate if out_pregate is not None else out,
            final_ssm_state,
            final_k_prev1_state,
            final_k_prev2_state,
            final_v_prev1_state,
            final_v_prev2_state,
            chunk_ssm_starts if chunk_ssm_starts is not None else final_ssm_state,
            chunk_k_prev1_starts if chunk_k_prev1_starts is not None else final_k_prev1_state,
            chunk_k_prev2_starts if chunk_k_prev2_starts is not None else final_k_prev2_state,
            chunk_v_prev1_starts if chunk_v_prev1_starts is not None else final_v_prev1_state,
            chunk_v_prev2_starts if chunk_v_prev2_starts is not None else final_v_prev2_state,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            ADT.stride(0),
            ADT.stride(1),
            ADT.stride(2),
            DT.stride(0),
            DT.stride(1),
            DT.stride(2),
            Simpson.stride(0),
            Simpson.stride(1),
            Simpson.stride(2),
            Midpoint.stride(0) if Midpoint is not None else 0,
            Midpoint.stride(1) if Midpoint is not None else 0,
            Midpoint.stride(2) if Midpoint is not None else 0,
            Q_bias.stride(0),
            Q_bias.stride(1),
            K_bias.stride(0),
            K_bias.stride(1),
            angles_cumsum.stride(0),
            angles_cumsum.stride(1),
            angles_cumsum.stride(2),
            angles_cumsum.stride(3),
            D.stride(0) if D is not None else 0,
            Z.stride(0) if Z is not None else 0,
            Z.stride(1) if Z is not None else 0,
            Z.stride(2) if Z is not None else 0,
            Z.stride(3) if Z is not None else 0,
            init_ssm_state.stride(0),
            init_ssm_state.stride(1),
            init_ssm_state.stride(2),
            init_ssm_state.stride(3),
            init_k_prev1_state.stride(0),
            init_k_prev1_state.stride(1),
            init_k_prev1_state.stride(2),
            init_k_prev2_state.stride(0),
            init_k_prev2_state.stride(1),
            init_k_prev2_state.stride(2),
            init_v_prev1_state.stride(0),
            init_v_prev1_state.stride(1),
            init_v_prev1_state.stride(2),
            init_v_prev2_state.stride(0),
            init_v_prev2_state.stride(1),
            init_v_prev2_state.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out_pregate.stride(0) if out_pregate is not None else 0,
            out_pregate.stride(1) if out_pregate is not None else 0,
            out_pregate.stride(2) if out_pregate is not None else 0,
            out_pregate.stride(3) if out_pregate is not None else 0,
            final_ssm_state.stride(0) if needs_final_state_buffers else 0,
            final_ssm_state.stride(1) if needs_final_state_buffers else 0,
            final_ssm_state.stride(2) if needs_final_state_buffers else 0,
            final_ssm_state.stride(3) if needs_final_state_buffers else 0,
            final_k_prev1_state.stride(0) if needs_final_state_buffers else 0,
            final_k_prev1_state.stride(1) if needs_final_state_buffers else 0,
            final_k_prev1_state.stride(2) if needs_final_state_buffers else 0,
            final_k_prev2_state.stride(0) if needs_final_state_buffers else 0,
            final_k_prev2_state.stride(1) if needs_final_state_buffers else 0,
            final_k_prev2_state.stride(2) if needs_final_state_buffers else 0,
            final_v_prev1_state.stride(0) if needs_final_state_buffers else 0,
            final_v_prev1_state.stride(1) if needs_final_state_buffers else 0,
            final_v_prev1_state.stride(2) if needs_final_state_buffers else 0,
            final_v_prev2_state.stride(0) if needs_final_state_buffers else 0,
            final_v_prev2_state.stride(1) if needs_final_state_buffers else 0,
            final_v_prev2_state.stride(2) if needs_final_state_buffers else 0,
            chunk_ssm_starts.stride(0) if chunk_ssm_starts is not None else 0,
            chunk_ssm_starts.stride(1) if chunk_ssm_starts is not None else 0,
            chunk_ssm_starts.stride(2) if chunk_ssm_starts is not None else 0,
            chunk_ssm_starts.stride(3) if chunk_ssm_starts is not None else 0,
            chunk_ssm_starts.stride(4) if chunk_ssm_starts is not None else 0,
            chunk_k_prev1_starts.stride(0) if chunk_k_prev1_starts is not None else 0,
            chunk_k_prev1_starts.stride(1) if chunk_k_prev1_starts is not None else 0,
            chunk_k_prev1_starts.stride(2) if chunk_k_prev1_starts is not None else 0,
            chunk_k_prev1_starts.stride(3) if chunk_k_prev1_starts is not None else 0,
            chunk_k_prev2_starts.stride(0) if chunk_k_prev2_starts is not None else 0,
            chunk_k_prev2_starts.stride(1) if chunk_k_prev2_starts is not None else 0,
            chunk_k_prev2_starts.stride(2) if chunk_k_prev2_starts is not None else 0,
            chunk_k_prev2_starts.stride(3) if chunk_k_prev2_starts is not None else 0,
            chunk_v_prev1_starts.stride(0) if chunk_v_prev1_starts is not None else 0,
            chunk_v_prev1_starts.stride(1) if chunk_v_prev1_starts is not None else 0,
            chunk_v_prev1_starts.stride(2) if chunk_v_prev1_starts is not None else 0,
            chunk_v_prev1_starts.stride(3) if chunk_v_prev1_starts is not None else 0,
            chunk_v_prev2_starts.stride(0) if chunk_v_prev2_starts is not None else 0,
            chunk_v_prev2_starts.stride(1) if chunk_v_prev2_starts is not None else 0,
            chunk_v_prev2_starts.stride(2) if chunk_v_prev2_starts is not None else 0,
            chunk_v_prev2_starts.stride(3) if chunk_v_prev2_starts is not None else 0,
            seqlen,
            nheads_qk,
            n_angles,
            CHUNK_SIZE=chunk_size,
            HEADDIM_QK=headdim_qk,
            HEADDIM_V=headdim_v,
            HAS_D=D is not None,
            HAS_Z=Z is not None,
            HAS_MIDPOINT=Midpoint is not None,
            HAS_INITIAL_STATES=Initial_States is not None,
            RETURN_FINAL_STATES=needs_final_state_buffers,
            STORE_CHUNK_BOUNDARY_STATES=store_states_adt_outv,
        )
        final_states = None
        if needs_final_state_buffers:
            final_states = (
                final_angle_state,
                final_ssm_state,
                final_k_prev1_state,
                final_k_prev2_state,
                final_v_prev1_state,
                final_v_prev2_state,
            )

    return (
        out,
        out_pregate,
        angles_cumsum if headdim_qk >= 16 else None,
        chunk_ssm_starts,
        chunk_k_prev1_starts,
        chunk_k_prev2_starts,
        chunk_v_prev1_starts,
        chunk_v_prev2_starts,
        final_states,
    )
