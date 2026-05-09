"""Forward-only fused Simamba SISO prefill kernel.

This experimental profiling kernel targets the fixed-length prefill case that
currently falls back to many tiny PyTorch elementwise launches in
``profiling/simamba``.  It intentionally lives under ``profiling/test_kernel``
so it can be benchmarked without changing the production module path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.triton.mamba3.utils import cos_approx, silu, sin_approx, tanh_approx


@triton.jit
def _simamba_siso_prefill_chunk_kernel(
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
    Angle_State,
    SSM_State,
    K_Prev1_State,
    K_Prev2_State,
    V_Prev1_State,
    V_Prev2_State,
    Out,
    seqlen,
    chunk_start,
    block_len,
    nheads_qk,
    headdim_angles,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MIDPOINT: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
    IS_FULL_CHUNK: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    offs_qk = tl.arange(0, HEADDIM_QK)
    offs_v = tl.arange(0, HEADDIM_V)
    offs_pair = tl.arange(0, HEADDIM_QK // 2)

    q_bias = tl.load(Q_bias + pid_head * HEADDIM_QK + offs_qk).to(tl.float32)
    k_bias = tl.load(K_bias + pid_head * HEADDIM_QK + offs_qk).to(tl.float32)
    d_val = tl.load(D + pid_head).to(tl.float32) if HAS_D else 0.0

    state_bh = pid_batch * nheads + pid_head
    angle_state_base = state_bh * (HEADDIM_QK // 2)
    ssm_state_base = state_bh * HEADDIM_V * HEADDIM_QK
    k_state_base = state_bh * HEADDIM_QK
    v_state_base = state_bh * HEADDIM_V

    if IS_FIRST_CHUNK:
        acc_ssm = tl.zeros((HEADDIM_V, HEADDIM_QK), dtype=tl.float32)
        k_prev1 = tl.zeros((HEADDIM_QK,), dtype=tl.float32)
        k_prev2 = tl.zeros((HEADDIM_QK,), dtype=tl.float32)
        v_prev1 = tl.zeros((HEADDIM_V,), dtype=tl.float32)
        v_prev2 = tl.zeros((HEADDIM_V,), dtype=tl.float32)
        angle_state = tl.zeros((HEADDIM_QK // 2,), dtype=tl.float32)
    else:
        acc_ssm = tl.load(
            SSM_State + ssm_state_base + offs_v[:, None] * HEADDIM_QK + offs_qk[None, :]
        ).to(tl.float32)
        k_prev1 = tl.load(K_Prev1_State + k_state_base + offs_qk).to(tl.float32)
        k_prev2 = tl.load(K_Prev2_State + k_state_base + offs_qk).to(tl.float32)
        v_prev1 = tl.load(V_Prev1_State + v_state_base + offs_v).to(tl.float32)
        v_prev2 = tl.load(V_Prev2_State + v_state_base + offs_v).to(tl.float32)
        angle_state = tl.load(Angle_State + angle_state_base + offs_pair).to(tl.float32)

    pi = 3.141592653589793
    two_pi = 6.283185307179586
    log2e = 1.44269504089
    half_log2e = 0.721347520445

    for step in range(0, BLOCK_SIZE):
        active = True if IS_FULL_CHUNK else step < block_len
        token_idx = chunk_start + step
        q_base = ((pid_batch * seqlen + token_idx) * nheads_qk + head_idx_qk) * HEADDIM_QK
        kv_base = ((pid_batch * seqlen + token_idx) * nheads + pid_head)
        v_base = kv_base * HEADDIM_V
        coeff_base = (pid_batch * nheads + pid_head) * seqlen + token_idx

        q_pre = tl.load(Q + q_base + offs_qk, mask=active, other=0.0).to(tl.float32) + q_bias
        k_pre = tl.load(K + q_base + offs_qk, mask=active, other=0.0).to(tl.float32) + k_bias
        v_t = tl.load(V + v_base + offs_v, mask=active, other=0.0).to(tl.float32)

        dt = tl.load(DT + coeff_base, mask=active, other=0.0).to(tl.float32)
        adt = tl.load(ADT + coeff_base, mask=active, other=0.0).to(tl.float32)
        simpson = tl.load(Simpson + coeff_base, mask=active, other=0.0).to(tl.float32)
        simpson = tl.minimum(1.0, tl.maximum(0.0, simpson))
        if HAS_MIDPOINT:
            midpoint = tl.load(Midpoint + coeff_base, mask=active, other=0.0).to(tl.float32)
            midpoint = tl.minimum(1.0, tl.maximum(0.0, midpoint))
            simpson = tl.minimum(1.0, tl.maximum(0.0, simpson * midpoint))

        angle_base = kv_base * headdim_angles
        angle_raw = tl.load(
            Angles + angle_base + offs_pair,
            mask=active & (offs_pair < headdim_angles),
            other=0.0,
        ).to(tl.float32)
        angle_state += tanh_approx(angle_raw) * dt * pi
        angle_state = angle_state - two_pi * tl.floor(angle_state / two_pi)

        cos_block = cos_approx(angle_state)
        sin_block = sin_approx(angle_state)

        q0, q1 = tl.split(tl.reshape(q_pre, (HEADDIM_QK // 2, 2)))
        q_rot0 = q0 * cos_block - q1 * sin_block
        q_rot1 = q0 * sin_block + q1 * cos_block
        q_rot = tl.reshape(tl.join(q_rot0, q_rot1), (HEADDIM_QK,))

        k0, k1 = tl.split(tl.reshape(k_pre, (HEADDIM_QK // 2, 2)))
        k_rot0 = k0 * cos_block - k1 * sin_block
        k_rot1 = k0 * sin_block + k1 * cos_block
        k_rot = tl.reshape(tl.join(k_rot0, k_rot1), (HEADDIM_QK,))

        alpha = tl.math.exp2(adt * log2e)
        alpha_half = tl.math.exp2(adt * half_log2e)
        gamma0 = (dt / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * simpson))
        gamma1 = (dt / 6.0) * (alpha + alpha_half * (2.0 + simpson))
        gamma2 = -(dt / 12.0) * alpha_half * simpson

        kv_t = v_t[:, None] * k_rot[None, :]
        kv_prev1 = v_prev1[:, None] * k_prev1[None, :]
        kv_prev2 = v_prev2[:, None] * k_prev2[None, :]
        next_ssm = alpha * acc_ssm + gamma0 * kv_t + gamma1 * kv_prev1 + gamma2 * kv_prev2
        acc_ssm = tl.where(active, next_ssm, acc_ssm)

        out_t = tl.sum(acc_ssm * q_rot[None, :], axis=1)
        if HAS_D:
            out_t += d_val * v_t
        if HAS_Z:
            z_t = tl.load(Z + v_base + offs_v, mask=active, other=0.0).to(tl.float32)
            out_t *= silu(z_t)

        tl.store(Out + v_base + offs_v, out_t, mask=active)

        old_k_prev1 = k_prev1
        old_v_prev1 = v_prev1
        k_prev1 = tl.where(active, k_rot, k_prev1)
        k_prev2 = tl.where(active, old_k_prev1, k_prev2)
        v_prev1 = tl.where(active, v_t, v_prev1)
        v_prev2 = tl.where(active, old_v_prev1, v_prev2)

    tl.store(Angle_State + angle_state_base + offs_pair, angle_state)
    tl.store(
        SSM_State + ssm_state_base + offs_v[:, None] * HEADDIM_QK + offs_qk[None, :],
        acc_ssm,
    )
    tl.store(K_Prev1_State + k_state_base + offs_qk, k_prev1)
    tl.store(K_Prev2_State + k_state_base + offs_qk, k_prev2)
    tl.store(V_Prev1_State + v_state_base + offs_v, v_prev1)
    tl.store(V_Prev2_State + v_state_base + offs_v, v_prev2)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def improved_simamba_siso_forward(
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
    chunk_size: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run the fused fixed-length Simamba prefill kernel.

    The prototype supports the profiling path: no variable-length batches, no
    incoming recurrent state, and forward output only.  It launches one fused
    Triton kernel per ``chunk_size`` tokens and carries the recurrent state in
    compact workspace buffers between chunks.
    """
    if not Q.is_cuda:
        raise RuntimeError("improved_simamba_siso_forward requires CUDA tensors")

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    if K.shape != Q.shape:
        raise ValueError(f"Q/K shape mismatch: {Q.shape} vs {K.shape}")
    if headdim_qk % 2 != 0 or not _is_power_of_two(headdim_qk):
        raise ValueError(f"headdim_qk must be an even power of two, got {headdim_qk}")

    if V.dim() != 4:
        raise ValueError(f"Expected V to be 4D, got {V.shape}")
    if V.shape[0] != batch or V.shape[1] != seqlen:
        raise ValueError(f"V batch/seqlen mismatch: expected {(batch, seqlen)}, got {V.shape[:2]}")
    nheads = V.shape[2]
    headdim_v = V.shape[3]
    if nheads % nheads_qk != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})")
    if not _is_power_of_two(headdim_v):
        raise ValueError(f"headdim_v must be a power of two, got {headdim_v}")
    if chunk_size <= 0 or not _is_power_of_two(chunk_size):
        raise ValueError(f"chunk_size must be a positive power of two, got {chunk_size}")
    if num_warps not in {1, 2, 4, 8, 16, 32}:
        raise ValueError(f"num_warps must be a supported Triton warp count, got {num_warps}")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")

    if ADT.shape != (batch, nheads, seqlen):
        raise ValueError(f"ADT shape mismatch: expected {(batch, nheads, seqlen)}, got {ADT.shape}")
    if DT.shape != (batch, nheads, seqlen):
        raise ValueError(f"DT shape mismatch: expected {(batch, nheads, seqlen)}, got {DT.shape}")
    if Simpson.shape != (batch, nheads, seqlen):
        raise ValueError(f"Simpson shape mismatch: expected {(batch, nheads, seqlen)}, got {Simpson.shape}")
    if Midpoint is not None and Midpoint.shape != (batch, nheads, seqlen):
        raise ValueError(f"Midpoint shape mismatch: expected {(batch, nheads, seqlen)}, got {Midpoint.shape}")
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"Q_bias shape mismatch: expected {(nheads, headdim_qk)}, got {Q_bias.shape}")
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"K_bias shape mismatch: expected {(nheads, headdim_qk)}, got {K_bias.shape}")

    headdim_angles = Angles.shape[-1]
    if Angles.shape != (batch, seqlen, nheads, headdim_angles):
        raise ValueError(
            f"Angles shape mismatch: expected {(batch, seqlen, nheads, headdim_angles)}, got {Angles.shape}"
        )
    if headdim_angles > headdim_qk // 2 or headdim_angles < 0:
        raise ValueError(f"Invalid rotary angle dim {headdim_angles} for headdim_qk={headdim_qk}")
    if D is not None and D.shape != (nheads,):
        raise ValueError(f"D shape mismatch: expected {(nheads,)}, got {D.shape}")
    if Z is not None and Z.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(f"Z shape mismatch: expected {(batch, seqlen, nheads, headdim_v)}, got {Z.shape}")

    has_midpoint = Midpoint is not None
    has_d = D is not None
    has_z = Z is not None

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    ADT = ADT.contiguous()
    DT = DT.contiguous()
    Simpson = Simpson.contiguous()
    Midpoint = Midpoint.contiguous() if Midpoint is not None else Simpson
    Q_bias = Q_bias.contiguous()
    K_bias = K_bias.contiguous()
    Angles = Angles.contiguous()
    D = D.contiguous() if D is not None else Q_bias.new_empty((1,))
    Z = Z.contiguous() if Z is not None else V

    out = torch.empty((batch, seqlen, nheads, headdim_v), device=V.device, dtype=V.dtype)
    angle_state = torch.empty((batch, nheads, headdim_qk // 2), device=Q.device, dtype=torch.float32)
    ssm_state = torch.empty((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    k_prev1_state = torch.empty((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
    k_prev2_state = torch.empty_like(k_prev1_state)
    v_prev1_state = torch.empty((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    v_prev2_state = torch.empty_like(v_prev1_state)

    for start in range(0, seqlen, chunk_size):
        block_len = min(chunk_size, seqlen - start)
        _simamba_siso_prefill_chunk_kernel[(nheads, batch)](
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
            angle_state,
            ssm_state,
            k_prev1_state,
            k_prev2_state,
            v_prev1_state,
            v_prev2_state,
            out,
            seqlen,
            start,
            block_len,
            nheads_qk,
            headdim_angles,
            HEADDIM_QK=headdim_qk,
            HEADDIM_V=headdim_v,
            BLOCK_SIZE=chunk_size,
            HAS_MIDPOINT=has_midpoint,
            HAS_D=has_d,
            HAS_Z=has_z,
            IS_FIRST_CHUNK=start == 0,
            IS_FULL_CHUNK=block_len == chunk_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
