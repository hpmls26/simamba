"""Simamba SISO Triton step kernel.

This kernel implements the locked Simpson recurrence:
    h_t = alpha_t * h_{t-1}
        + gamma0_t * kv_t
        + gamma1_t * kv_{t-1}
        + gamma2_t * kv_{t-2}
"""

from typing import Optional, Tuple

import torch

import triton
import triton.language as tl
from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx, silu, tanh_approx


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["HEADDIM_QK", "HEADDIM_V", "HAS_D", "HAS_Z", "HAS_MIDPOINT"],
)
@triton.jit
def mamba3_siso_step_kernel(
    # Inputs
    Q,
    K,
    V,
    ADT,
    DT,
    Simpson,
    Midpoint,
    Q_bias,
    K_bias,
    Angles,
    D,
    Z,
    Input_Angle_State,
    Input_SSM_State,
    Input_K_prev1_State,
    Input_K_prev2_State,
    Input_V_prev1_State,
    Input_V_prev2_State,
    # Outputs
    Out,
    Output_Angle_State,
    Output_SSM_State,
    Output_K_prev1_State,
    Output_K_prev2_State,
    Output_V_prev1_State,
    Output_V_prev2_State,
    # Input strides
    stride_q_batch,
    stride_q_head,
    stride_q_qkdim,
    stride_k_batch,
    stride_k_head,
    stride_k_qkdim,
    stride_v_batch,
    stride_v_head,
    stride_v_vdim,
    stride_adt_batch,
    stride_adt_head,
    stride_dt_batch,
    stride_dt_head,
    stride_simpson_batch,
    stride_simpson_head,
    stride_midpoint_batch,
    stride_midpoint_head,
    stride_q_bias_head,
    stride_q_bias_qkdim,
    stride_k_bias_head,
    stride_k_bias_qkdim,
    stride_angles_batch,
    stride_angles_head,
    stride_angles_qkdim,
    stride_d_head,
    stride_z_batch,
    stride_z_head,
    stride_z_vdim,
    stride_angle_state_batch,
    stride_angle_state_head,
    stride_angle_state_anglesdim,
    stride_input_ssm_state_batch,
    stride_input_ssm_state_head,
    stride_input_ssm_state_vdim,
    stride_input_ssm_state_qkdim,
    stride_input_k_prev1_state_batch,
    stride_input_k_prev1_state_head,
    stride_input_k_prev1_state_qkdim,
    stride_input_k_prev2_state_batch,
    stride_input_k_prev2_state_head,
    stride_input_k_prev2_state_qkdim,
    stride_input_v_prev1_state_batch,
    stride_input_v_prev1_state_head,
    stride_input_v_prev1_state_vdim,
    stride_input_v_prev2_state_batch,
    stride_input_v_prev2_state_head,
    stride_input_v_prev2_state_vdim,
    # Output strides
    stride_o_batch,
    stride_o_head,
    stride_o_vdim,
    stride_output_angle_state_batch,
    stride_output_angle_state_head,
    stride_output_angle_state_anglesdim,
    stride_output_ssm_state_batch,
    stride_output_ssm_state_head,
    stride_output_ssm_state_vdim,
    stride_output_ssm_state_qkdim,
    stride_output_k_prev1_state_batch,
    stride_output_k_prev1_state_head,
    stride_output_k_prev1_state_qkdim,
    stride_output_k_prev2_state_batch,
    stride_output_k_prev2_state_head,
    stride_output_k_prev2_state_qkdim,
    stride_output_v_prev1_state_batch,
    stride_output_v_prev1_state_head,
    stride_output_v_prev1_state_vdim,
    stride_output_v_prev2_state_batch,
    stride_output_v_prev2_state_head,
    stride_output_v_prev2_state_vdim,
    # Dimensions
    nheads_qk,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    HEADDIM_ANGLES: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_MIDPOINT: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    q_ptr = Q + pid_batch * stride_q_batch + head_idx_qk * stride_q_head
    k_ptr = K + pid_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_ptr = V + pid_batch * stride_v_batch + pid_head * stride_v_head
    adt_ptr = ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head
    dt_ptr = DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
    simpson_ptr = Simpson + pid_batch * stride_simpson_batch + pid_head * stride_simpson_head
    midpoint_ptr = Midpoint + pid_batch * stride_midpoint_batch + pid_head * stride_midpoint_head

    q_bias_ptr = Q_bias + pid_head * stride_q_bias_head
    k_bias_ptr = K_bias + pid_head * stride_k_bias_head
    angle_ptr = Angles + pid_batch * stride_angles_batch + pid_head * stride_angles_head

    if HAS_D:
        D_ptr = D + pid_head * stride_d_head
        D_val = tl.load(D_ptr).to(tl.float32)
    if HAS_Z:
        z_ptr = Z + pid_batch * stride_z_batch + pid_head * stride_z_head

    input_angle_state_ptr = (
        Input_Angle_State + pid_batch * stride_angle_state_batch + pid_head * stride_angle_state_head
    )
    input_ssm_state_ptr = (
        Input_SSM_State + pid_batch * stride_input_ssm_state_batch + pid_head * stride_input_ssm_state_head
    )
    input_k_prev1_state_ptr = (
        Input_K_prev1_State
        + pid_batch * stride_input_k_prev1_state_batch
        + pid_head * stride_input_k_prev1_state_head
    )
    input_k_prev2_state_ptr = (
        Input_K_prev2_State
        + pid_batch * stride_input_k_prev2_state_batch
        + pid_head * stride_input_k_prev2_state_head
    )
    input_v_prev1_state_ptr = (
        Input_V_prev1_State
        + pid_batch * stride_input_v_prev1_state_batch
        + pid_head * stride_input_v_prev1_state_head
    )
    input_v_prev2_state_ptr = (
        Input_V_prev2_State
        + pid_batch * stride_input_v_prev2_state_batch
        + pid_head * stride_input_v_prev2_state_head
    )

    o_ptr = Out + pid_batch * stride_o_batch + pid_head * stride_o_head
    output_angle_state_ptr = (
        Output_Angle_State
        + pid_batch * stride_output_angle_state_batch
        + pid_head * stride_output_angle_state_head
    )
    output_ssm_state_ptr = (
        Output_SSM_State
        + pid_batch * stride_output_ssm_state_batch
        + pid_head * stride_output_ssm_state_head
    )
    output_k_prev1_state_ptr = (
        Output_K_prev1_State
        + pid_batch * stride_output_k_prev1_state_batch
        + pid_head * stride_output_k_prev1_state_head
    )
    output_k_prev2_state_ptr = (
        Output_K_prev2_State
        + pid_batch * stride_output_k_prev2_state_batch
        + pid_head * stride_output_k_prev2_state_head
    )
    output_v_prev1_state_ptr = (
        Output_V_prev1_State
        + pid_batch * stride_output_v_prev1_state_batch
        + pid_head * stride_output_v_prev1_state_head
    )
    output_v_prev2_state_ptr = (
        Output_V_prev2_State
        + pid_batch * stride_output_v_prev2_state_batch
        + pid_head * stride_output_v_prev2_state_head
    )

    PI = 3.141592653589793
    TWO_PI = 2 * PI
    offs_qk = tl.arange(0, HEADDIM_QK)
    offs_v = tl.arange(0, HEADDIM_V)
    offs_qkr = tl.arange(0, HEADDIM_QK // 2)

    q_pre_block = tl.load(q_ptr + offs_qk * stride_q_qkdim)
    k_pre_block = tl.load(k_ptr + offs_qk * stride_k_qkdim)

    q_bias_block = tl.load(q_bias_ptr + offs_qk * stride_q_bias_qkdim)
    k_bias_block = tl.load(k_bias_ptr + offs_qk * stride_k_bias_qkdim)
    q_pre_block += q_bias_block
    k_pre_block += k_bias_block

    dt = tl.load(dt_ptr).to(tl.float32)
    angle_block = tl.load(
        angle_ptr + offs_qkr * stride_angles_qkdim,
        mask=offs_qkr < HEADDIM_ANGLES,
        other=0.0,
    )
    angle_block = tanh_approx(angle_block.to(tl.float32)) * PI * dt
    angle_state = tl.load(
        input_angle_state_ptr + offs_qkr * stride_angle_state_anglesdim,
        mask=offs_qkr < HEADDIM_ANGLES,
        other=0.0,
    )

    angle_block += angle_state
    angle_block -= TWO_PI * tl.floor(angle_block / TWO_PI)
    tl.store(
        output_angle_state_ptr + offs_qkr * stride_output_angle_state_anglesdim,
        angle_block,
        mask=offs_qkr < HEADDIM_ANGLES,
    )

    cos_block = cos_approx(angle_block.to(tl.float32))
    sin_block = sin_approx(angle_block.to(tl.float32))

    q0, q1 = tl.split(tl.reshape(q_pre_block, [HEADDIM_QK // 2, 2]))
    qo0 = q0 * cos_block - q1 * sin_block
    qo1 = q0 * sin_block + q1 * cos_block
    q_block = tl.reshape(tl.join(qo0, qo1), [HEADDIM_QK]).to(q_pre_block.dtype)

    k0, k1 = tl.split(tl.reshape(k_pre_block, [HEADDIM_QK // 2, 2]))
    ko0 = k0 * cos_block - k1 * sin_block
    ko1 = k0 * sin_block + k1 * cos_block
    k_block = tl.reshape(tl.join(ko0, ko1), [HEADDIM_QK]).to(k_pre_block.dtype)

    v_block = tl.load(v_ptr + offs_v * stride_v_vdim)

    k_prev1_state = tl.load(input_k_prev1_state_ptr + offs_qk * stride_input_k_prev1_state_qkdim)
    k_prev2_state = tl.load(input_k_prev2_state_ptr + offs_qk * stride_input_k_prev2_state_qkdim)
    v_prev1_state = tl.load(input_v_prev1_state_ptr + offs_v * stride_input_v_prev1_state_vdim)
    v_prev2_state = tl.load(input_v_prev2_state_ptr + offs_v * stride_input_v_prev2_state_vdim)

    adt = tl.load(adt_ptr).to(tl.float32)
    simpson = tl.load(simpson_ptr).to(tl.float32)
    simpson = tl.minimum(1.0, tl.maximum(0.0, simpson))
    if HAS_MIDPOINT:
        midpoint = tl.load(midpoint_ptr).to(tl.float32)
        midpoint = tl.minimum(1.0, tl.maximum(0.0, midpoint))
        simpson = tl.minimum(1.0, tl.maximum(0.0, simpson * midpoint))

    alpha = tl.math.exp2(adt * 1.44269504089)
    alpha_half = tl.math.exp2(adt * 0.721347520445)
    gamma0 = (dt / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * simpson))
    gamma1 = (dt / 6.0) * (alpha + alpha_half * (2.0 + simpson))
    gamma2 = -(dt / 12.0) * alpha_half * simpson

    ssm_state = tl.load(
        input_ssm_state_ptr
        + offs_v[:, None] * stride_input_ssm_state_vdim
        + offs_qk[None, :] * stride_input_ssm_state_qkdim
    ).to(tl.float32)

    kv_t = v_block[:, None] * k_block[None, :]
    kv_prev1 = v_prev1_state[:, None] * k_prev1_state[None, :]
    kv_prev2 = v_prev2_state[:, None] * k_prev2_state[None, :]

    ssm_state = (
        alpha * ssm_state
        + gamma0 * kv_t
        + gamma1 * kv_prev1
        + gamma2 * kv_prev2
    )

    tl.store(
        output_ssm_state_ptr
        + offs_v[:, None] * stride_output_ssm_state_vdim
        + offs_qk[None, :] * stride_output_ssm_state_qkdim,
        ssm_state,
    )

    # Use an elementwise reduction instead of tl.dot so small test dims
    # (e.g., headdim_qk=8) compile without Triton MMA tile constraints.
    out = tl.sum(ssm_state * q_block.to(tl.float32)[None, :], axis=1)

    if HAS_D:
        out += D_val * v_block
    if HAS_Z:
        z_block = tl.load(z_ptr + offs_v * stride_z_vdim)
        out = out * silu(z_block.to(tl.float32))

    tl.store(o_ptr + offs_v * stride_o_vdim, out)

    tl.store(output_k_prev1_state_ptr + offs_qk * stride_output_k_prev1_state_qkdim, k_block)
    tl.store(output_k_prev2_state_ptr + offs_qk * stride_output_k_prev2_state_qkdim, k_prev1_state)
    tl.store(output_v_prev1_state_ptr + offs_v * stride_output_v_prev1_state_vdim, v_block)
    tl.store(output_v_prev2_state_ptr + offs_v * stride_output_v_prev2_state_vdim, v_prev1_state)


def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Custom allocator for Triton runtime allocations."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def mamba3_siso_step(
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
    Out: Optional[torch.Tensor] = None,
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
    Output_States: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = None,
):
    """Simamba step wrapper with 6-state recurrence cache."""
    batch, nheads_qk, headdim_qk = Q.shape
    _, nheads, headdim_v = V.shape
    device = Q.device

    assert Q.shape == K.shape, f"Q and K shape mismatch: {Q.shape} vs {K.shape}"
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert ADT.shape == (batch, nheads), f"ADT shape mismatch: expected {(batch, nheads)}, got {ADT.shape}"
    assert DT.shape == (batch, nheads), f"DT shape mismatch: expected {(batch, nheads)}, got {DT.shape}"
    assert Simpson.shape == (batch, nheads), (
        f"Simpson shape mismatch: expected {(batch, nheads)}, got {Simpson.shape}"
    )
    if Midpoint is not None:
        assert Midpoint.shape == (batch, nheads), (
            f"Midpoint shape mismatch: expected {(batch, nheads)}, got {Midpoint.shape}"
        )

    assert Q_bias.shape == (nheads, headdim_qk)
    assert K_bias.shape == (nheads, headdim_qk)

    headdim_angles = Angles.shape[-1]
    assert headdim_angles <= headdim_qk // 2 and headdim_angles % 2 == 0
    assert Angles.shape == (batch, nheads, headdim_angles)

    if D is not None:
        assert D.shape == (nheads,)
    if Z is not None:
        assert Z.shape == (batch, nheads, headdim_v)

    if Input_States is None:
        Input_Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        Input_SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        Input_K_prev1_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        Input_K_prev2_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        Input_V_prev1_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
        Input_V_prev2_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
    else:
        (
            Input_Angle_State,
            Input_SSM_State,
            Input_K_prev1_State,
            Input_K_prev2_State,
            Input_V_prev1_State,
            Input_V_prev2_State,
        ) = Input_States

    Q = Q.contiguous() if not Q.is_contiguous() else Q
    K = K.contiguous() if not K.is_contiguous() else K
    V = V.contiguous() if not V.is_contiguous() else V
    ADT = ADT.contiguous() if not ADT.is_contiguous() else ADT
    DT = DT.contiguous() if not DT.is_contiguous() else DT
    Simpson = Simpson.contiguous() if not Simpson.is_contiguous() else Simpson
    if Midpoint is not None:
        Midpoint = Midpoint.contiguous() if not Midpoint.is_contiguous() else Midpoint
    Q_bias = Q_bias.contiguous() if not Q_bias.is_contiguous() else Q_bias
    K_bias = K_bias.contiguous() if not K_bias.is_contiguous() else K_bias
    Angles = Angles.contiguous() if not Angles.is_contiguous() else Angles
    if D is not None:
        D = D.contiguous() if not D.is_contiguous() else D
    if Z is not None:
        Z = Z.contiguous() if not Z.is_contiguous() else Z

    if Out is None:
        Out = torch.empty((batch, nheads, headdim_v), device=device, dtype=V.dtype)
    else:
        assert Out.shape == (batch, nheads, headdim_v)

    if Output_States is None:
        Output_Angle_State = torch.empty((batch, nheads, headdim_angles), device=device, dtype=torch.float32)
        Output_SSM_State = torch.empty((batch, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)
        Output_K_prev1_State = torch.empty((batch, nheads, headdim_qk), device=device, dtype=Q.dtype)
        Output_K_prev2_State = torch.empty((batch, nheads, headdim_qk), device=device, dtype=Q.dtype)
        Output_V_prev1_State = torch.empty((batch, nheads, headdim_v), device=device, dtype=V.dtype)
        Output_V_prev2_State = torch.empty((batch, nheads, headdim_v), device=device, dtype=V.dtype)
    else:
        (
            Output_Angle_State,
            Output_SSM_State,
            Output_K_prev1_State,
            Output_K_prev2_State,
            Output_V_prev1_State,
            Output_V_prev2_State,
        ) = Output_States
        assert Output_Angle_State.shape == (batch, nheads, headdim_angles)
        assert Output_SSM_State.shape == (batch, nheads, headdim_v, headdim_qk)
        assert Output_K_prev1_State.shape == (batch, nheads, headdim_qk)
        assert Output_K_prev2_State.shape == (batch, nheads, headdim_qk)
        assert Output_V_prev1_State.shape == (batch, nheads, headdim_v)
        assert Output_V_prev2_State.shape == (batch, nheads, headdim_v)

    midpoint_ptr = Midpoint if Midpoint is not None else Simpson
    grid = (nheads, batch)
    mamba3_siso_step_kernel[grid](
        Q,
        K,
        V,
        ADT,
        DT,
        Simpson,
        midpoint_ptr,
        Q_bias,
        K_bias,
        Angles,
        D,
        Z,
        Input_Angle_State,
        Input_SSM_State,
        Input_K_prev1_State,
        Input_K_prev2_State,
        Input_V_prev1_State,
        Input_V_prev2_State,
        Out,
        Output_Angle_State,
        Output_SSM_State,
        Output_K_prev1_State,
        Output_K_prev2_State,
        Output_V_prev1_State,
        Output_V_prev2_State,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        ADT.stride(0),
        ADT.stride(1),
        DT.stride(0),
        DT.stride(1),
        Simpson.stride(0),
        Simpson.stride(1),
        Midpoint.stride(0) if Midpoint is not None else 0,
        Midpoint.stride(1) if Midpoint is not None else 0,
        Q_bias.stride(0),
        Q_bias.stride(1),
        K_bias.stride(0),
        K_bias.stride(1),
        Angles.stride(0),
        Angles.stride(1),
        Angles.stride(2),
        D.stride(0) if D is not None else 0,
        Z.stride(0) if Z is not None else 0,
        Z.stride(1) if Z is not None else 0,
        Z.stride(2) if Z is not None else 0,
        Input_Angle_State.stride(0),
        Input_Angle_State.stride(1),
        Input_Angle_State.stride(2),
        Input_SSM_State.stride(0),
        Input_SSM_State.stride(1),
        Input_SSM_State.stride(2),
        Input_SSM_State.stride(3),
        Input_K_prev1_State.stride(0),
        Input_K_prev1_State.stride(1),
        Input_K_prev1_State.stride(2),
        Input_K_prev2_State.stride(0),
        Input_K_prev2_State.stride(1),
        Input_K_prev2_State.stride(2),
        Input_V_prev1_State.stride(0),
        Input_V_prev1_State.stride(1),
        Input_V_prev1_State.stride(2),
        Input_V_prev2_State.stride(0),
        Input_V_prev2_State.stride(1),
        Input_V_prev2_State.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Output_Angle_State.stride(0),
        Output_Angle_State.stride(1),
        Output_Angle_State.stride(2),
        Output_SSM_State.stride(0),
        Output_SSM_State.stride(1),
        Output_SSM_State.stride(2),
        Output_SSM_State.stride(3),
        Output_K_prev1_State.stride(0),
        Output_K_prev1_State.stride(1),
        Output_K_prev1_State.stride(2),
        Output_K_prev2_State.stride(0),
        Output_K_prev2_State.stride(1),
        Output_K_prev2_State.stride(2),
        Output_V_prev1_State.stride(0),
        Output_V_prev1_State.stride(1),
        Output_V_prev1_State.stride(2),
        Output_V_prev2_State.stride(0),
        Output_V_prev2_State.stride(1),
        Output_V_prev2_State.stride(2),
        nheads_qk,
        headdim_qk,
        headdim_v,
        headdim_angles,
        HAS_D=D is not None,
        HAS_Z=Z is not None,
        HAS_MIDPOINT=Midpoint is not None,
    )

    Output_States = [
        Output_Angle_State,
        Output_SSM_State,
        Output_K_prev1_State,
        Output_K_prev2_State,
        Output_V_prev1_State,
        Output_V_prev2_State,
    ]
    return Out, Output_States
