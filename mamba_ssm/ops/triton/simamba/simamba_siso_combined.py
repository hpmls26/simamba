"""Reference Simamba SISO implementation.

Phase A focuses on a numerically clear PyTorch path for Simpson-inspired
discretization with a width-3 state-input convolution.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


SIMAMBA_BOUNDARY_MODE_ZERO_PAD = "zero_pad"
SIMAMBA_SUPPORTED_BOUNDARY_MODES = (SIMAMBA_BOUNDARY_MODE_ZERO_PAD,)


def _validate_boundary_mode(boundary_mode: str) -> None:
    if boundary_mode not in SIMAMBA_SUPPORTED_BOUNDARY_MODES:
        raise ValueError(
            f"Unsupported boundary_mode={boundary_mode!r}. "
            f"Expected one of {SIMAMBA_SUPPORTED_BOUNDARY_MODES}."
        )


def _resolve_simpson_effective(simpson_t: Tensor, midpoint_t: Optional[Tensor]) -> Tensor:
    """Compute effective Simpson coefficient.

    midpoint_t is optional so parameter growth can stay disabled by default.
    When midpoint_t is absent, behavior is unchanged.
    """
    simpson = simpson_t.float().clamp(0.0, 1.0)
    if midpoint_t is None:
        return simpson
    midpoint = midpoint_t.float().clamp(0.0, 1.0)
    return (simpson * midpoint).clamp(0.0, 1.0)


def _apply_pairwise_rotary(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate q/k pairs using per-pair angles.

    Args:
        x: Tensor shaped (..., dim), dim must be even.
        angles: Tensor shaped (..., n_angles), n_angles <= dim // 2.
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Expected even head dim, got {dim}.")
    half = dim // 2
    n_angles = angles.shape[-1]
    if n_angles > half:
        raise ValueError(f"n_angles ({n_angles}) must be <= dim//2 ({half}).")

    x_pair = x.float().reshape(*x.shape[:-1], half, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]

    cos = torch.cos(angles.float())
    sin = torch.sin(angles.float())
    if n_angles < half:
        pad = half - n_angles
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    y = torch.stack((y0, y1), dim=-1).reshape(*x.shape[:-1], dim)
    return y.to(x.dtype)


def _compute_angle_cumsum(angles: Tensor, dt: Tensor, init_state: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """Compute cumulative rotary angles with optional initial state.

    Args:
        angles: (batch, seqlen, nheads, n_angles)
        dt: (batch, nheads, seqlen)
        init_state: (batch, nheads, n_angles) or None
    """
    dt_seq = dt.transpose(-1, -2).unsqueeze(-1).float()  # (b, l, h, 1)
    increments = torch.tanh(angles.float()) * dt_seq * math.pi
    cumsum = torch.cumsum(increments, dim=1)
    if init_state is not None:
        cumsum = cumsum + init_state.float().unsqueeze(1)
    two_pi = 2.0 * math.pi
    cumsum = torch.remainder(cumsum, two_pi)
    final_state = cumsum[:, -1]
    return cumsum, final_state


def _simpson_state_update(
    ssm_state: Tensor,
    kv_t: Tensor,
    kv_prev1: Tensor,
    kv_prev2: Tensor,
    adt_t: Tensor,
    dt_t: Tensor,
    simpson_t: Tensor,
    midpoint_t: Optional[Tensor] = None,
) -> Tensor:
    """Single-token Simpson-inspired state update.

    Recurrence is locked as:
        h_t = alpha_t h_{t-1}
            + gamma0_t * kv_t
            + gamma1_t * kv_{t-1}
            + gamma2_t * kv_{t-2}

    with
        s_eff_t = s_t, if midpoint_t is None
        s_eff_t = s_t * m_t, otherwise
        alpha_t = exp(adt_t)
        gamma0_t = dt_t / 6 * (1 + exp(adt_t / 2) * (2 - s_eff_t / 2))
        gamma1_t = dt_t / 6 * (exp(adt_t) + exp(adt_t / 2) * (2 + s_eff_t))
        gamma2_t = -dt_t / 12 * exp(adt_t / 2) * s_eff_t

    This implements a width-3 data-dependent convolution on kv terms by
    mixing kv_t, kv_{t-1}, and kv_{t-2} inside the recurrence.
    """
    alpha = torch.exp(adt_t.float())
    alpha_half = torch.exp(0.5 * adt_t.float())
    dt = dt_t.float()
    simpson_eff = _resolve_simpson_effective(simpson_t, midpoint_t)

    gamma0 = (dt / 6.0) * (1.0 + alpha_half * (2.0 - 0.5 * simpson_eff))
    gamma1 = (dt / 6.0) * (alpha + alpha_half * (2.0 + simpson_eff))
    gamma2 = -(dt / 12.0) * alpha_half * simpson_eff

    return (
        alpha[:, :, None, None] * ssm_state
        + gamma0[:, :, None, None] * kv_t
        + gamma1[:, :, None, None] * kv_prev1
        + gamma2[:, :, None, None] * kv_prev2
    )


def simamba_siso_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Simpson: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    Midpoint: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = None,
    chunk_size: int = 64,
    return_final_states: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    boundary_mode: str = SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reference Simamba SISO combined pass.

    Shapes:
        Q/K: (batch, seqlen, nheads, headdim_qk)
        V: (batch, seqlen, nheads, headdim_v)
        ADT/DT/Simpson: (batch, nheads, seqlen)
        Midpoint (optional): (batch, nheads, seqlen)
        Angles: (batch, seqlen, nheads, n_angles)
    """
    del chunk_size  # Kept for API compatibility.
    _validate_boundary_mode(boundary_mode)
    if cu_seqlens is not None:
        raise NotImplementedError("Phase A Simamba reference path does not support varlen yet.")

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    if K.shape != Q.shape:
        raise ValueError(f"Q and K shape mismatch: {Q.shape} vs {K.shape}.")
    if headdim_qk % 2 != 0:
        raise ValueError(f"headdim_qk must be even, got {headdim_qk}.")

    _, _, nheads, headdim_v = V.shape
    if nheads % nheads_qk != 0:
        raise ValueError(
            f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})."
        )
    if ADT.shape != (batch, nheads, seqlen):
        raise ValueError(f"ADT shape mismatch: got {ADT.shape}.")
    if DT.shape != (batch, nheads, seqlen):
        raise ValueError(f"DT shape mismatch: got {DT.shape}.")
    if Simpson.shape != (batch, nheads, seqlen):
        raise ValueError(f"Simpson shape mismatch: got {Simpson.shape}.")
    if Midpoint is not None and Midpoint.shape != (batch, nheads, seqlen):
        raise ValueError(f"Midpoint shape mismatch: got {Midpoint.shape}.")
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"Q_bias shape mismatch: got {Q_bias.shape}.")
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"K_bias shape mismatch: got {K_bias.shape}.")

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa_ratio, dim=2)
        K = K.repeat_interleave(gqa_ratio, dim=2)

    n_angles = Angles.shape[-1]
    if Angles.shape != (batch, seqlen, nheads, n_angles):
        raise ValueError(f"Angles shape mismatch: got {Angles.shape}.")
    if n_angles > headdim_qk // 2 or n_angles % 2 != 0:
        raise ValueError(
            f"Angles last dim must be even and <= headdim_qk//2, got {n_angles}."
        )

    if D is not None and D.shape != (nheads,):
        raise ValueError(f"D shape mismatch: got {D.shape}.")
    if Z is not None and Z.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(f"Z shape mismatch: got {Z.shape}.")

    if Input_States is not None:
        (
            input_angle_state,
            input_ssm_state,
            input_k_prev1,
            input_k_prev2,
            input_v_prev1,
            input_v_prev2,
        ) = Input_States
        if input_angle_state.shape != (batch, nheads, n_angles):
            raise ValueError(f"Input angle state shape mismatch: got {input_angle_state.shape}.")
        if input_ssm_state.shape != (batch, nheads, headdim_v, headdim_qk):
            raise ValueError(f"Input SSM state shape mismatch: got {input_ssm_state.shape}.")
        if input_k_prev1.shape != (batch, nheads, headdim_qk):
            raise ValueError(f"Input K prev1 shape mismatch: got {input_k_prev1.shape}.")
        if input_k_prev2.shape != (batch, nheads, headdim_qk):
            raise ValueError(f"Input K prev2 shape mismatch: got {input_k_prev2.shape}.")
        if input_v_prev1.shape != (batch, nheads, headdim_v):
            raise ValueError(f"Input V prev1 shape mismatch: got {input_v_prev1.shape}.")
        if input_v_prev2.shape != (batch, nheads, headdim_v):
            raise ValueError(f"Input V prev2 shape mismatch: got {input_v_prev2.shape}.")
    else:
        # Deterministic boundary rule:
        #   kv_{-1} = 0 and kv_{-2} = 0 (zero-padding history).
        # This fixes the first two-token behavior unambiguously.
        input_angle_state = None
        input_ssm_state = torch.zeros(
            (batch, nheads, headdim_v, headdim_qk),
            device=Q.device,
            dtype=torch.float32,
        )
        input_k_prev1 = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        input_k_prev2 = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        input_v_prev1 = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)
        input_v_prev2 = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)

    angles_cumsum, final_angle_state = _compute_angle_cumsum(
        Angles,
        DT,
        input_angle_state,
    )

    q_pre = Q + Q_bias[None, None, :, :]
    k_pre = K + K_bias[None, None, :, :]
    q_rot = _apply_pairwise_rotary(q_pre, angles_cumsum)
    k_rot = _apply_pairwise_rotary(k_pre, angles_cumsum)

    ssm_state = input_ssm_state.float()
    k_prev1 = input_k_prev1.float()
    k_prev2 = input_k_prev2.float()
    v_prev1 = input_v_prev1.float()
    v_prev2 = input_v_prev2.float()

    kv_prev1 = v_prev1.unsqueeze(-1) * k_prev1.unsqueeze(-2)
    kv_prev2 = v_prev2.unsqueeze(-1) * k_prev2.unsqueeze(-2)

    out = torch.empty((batch, seqlen, nheads, headdim_v), device=V.device, dtype=V.dtype)
    d_term = D.float()[None, :, None] if D is not None else None

    for t in range(seqlen):
        v_t = V[:, t].float()
        q_t = q_rot[:, t].float()
        k_t = k_rot[:, t].float()

        kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        ssm_state = _simpson_state_update(
            ssm_state,
            kv_t,
            kv_prev1,
            kv_prev2,
            ADT[:, :, t],
            DT[:, :, t],
            Simpson[:, :, t],
            Midpoint[:, :, t] if Midpoint is not None else None,
        )

        y_t = torch.einsum("bhpn,bhn->bhp", ssm_state, q_t)
        if d_term is not None:
            y_t = y_t + d_term * v_t
        if Z is not None:
            y_t = y_t * F.silu(Z[:, t].float())
        out[:, t] = y_t.to(V.dtype)

        kv_prev2 = kv_prev1
        kv_prev1 = kv_t
        k_prev2 = k_prev1
        k_prev1 = k_t
        v_prev2 = v_prev1
        v_prev1 = v_t

    if return_final_states:
        return (
            out,
            final_angle_state,
            ssm_state,
            k_prev1.to(K.dtype),
            k_prev2.to(K.dtype),
            v_prev1.to(V.dtype),
            v_prev2.to(V.dtype),
        )
    return out


def simamba_trapezoid_siso_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = None,
    return_final_states: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reference exponential-trapezoid SISO pass with the Simamba interface.

    This is a matched ablation baseline for Simamba: Q/K/V, biases, rotary
    angle dynamics, D skip, Z gating, and parameter shapes are identical, while
    the width-3 Simpson recurrence is replaced by the width-2 trapezoid update
    used by the Mamba-3 SISO formulation.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("Trapezoid reference path does not support varlen yet.")

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    if K.shape != Q.shape:
        raise ValueError(f"Q and K shape mismatch: {Q.shape} vs {K.shape}.")
    if headdim_qk % 2 != 0:
        raise ValueError(f"headdim_qk must be even, got {headdim_qk}.")

    _, _, nheads, headdim_v = V.shape
    if nheads % nheads_qk != 0:
        raise ValueError(
            f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})."
        )
    if ADT.shape != (batch, nheads, seqlen):
        raise ValueError(f"ADT shape mismatch: got {ADT.shape}.")
    if DT.shape != (batch, nheads, seqlen):
        raise ValueError(f"DT shape mismatch: got {DT.shape}.")
    if Trap.shape != (batch, nheads, seqlen):
        raise ValueError(f"Trap shape mismatch: got {Trap.shape}.")
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"Q_bias shape mismatch: got {Q_bias.shape}.")
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"K_bias shape mismatch: got {K_bias.shape}.")

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa_ratio, dim=2)
        K = K.repeat_interleave(gqa_ratio, dim=2)

    n_angles = Angles.shape[-1]
    if Angles.shape != (batch, seqlen, nheads, n_angles):
        raise ValueError(f"Angles shape mismatch: got {Angles.shape}.")
    if n_angles > headdim_qk // 2 or n_angles % 2 != 0:
        raise ValueError(
            f"Angles last dim must be even and <= headdim_qk//2, got {n_angles}."
        )

    if D is not None and D.shape != (nheads,):
        raise ValueError(f"D shape mismatch: got {D.shape}.")
    if Z is not None and Z.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(f"Z shape mismatch: got {Z.shape}.")

    if Input_States is not None:
        (
            input_angle_state,
            input_ssm_state,
            input_k_prev1,
            _input_k_prev2,
            input_v_prev1,
            _input_v_prev2,
        ) = Input_States
        if input_angle_state.shape != (batch, nheads, n_angles):
            raise ValueError(f"Input angle state shape mismatch: got {input_angle_state.shape}.")
        if input_ssm_state.shape != (batch, nheads, headdim_v, headdim_qk):
            raise ValueError(f"Input SSM state shape mismatch: got {input_ssm_state.shape}.")
        if input_k_prev1.shape != (batch, nheads, headdim_qk):
            raise ValueError(f"Input K prev1 shape mismatch: got {input_k_prev1.shape}.")
        if input_v_prev1.shape != (batch, nheads, headdim_v):
            raise ValueError(f"Input V prev1 shape mismatch: got {input_v_prev1.shape}.")
    else:
        input_angle_state = None
        input_ssm_state = torch.zeros(
            (batch, nheads, headdim_v, headdim_qk),
            device=Q.device,
            dtype=torch.float32,
        )
        input_k_prev1 = torch.zeros((batch, nheads, headdim_qk), device=Q.device, dtype=torch.float32)
        input_v_prev1 = torch.zeros((batch, nheads, headdim_v), device=Q.device, dtype=torch.float32)

    angles_cumsum, final_angle_state = _compute_angle_cumsum(
        Angles,
        DT,
        input_angle_state,
    )

    q_pre = Q + Q_bias[None, None, :, :]
    k_pre = K + K_bias[None, None, :, :]
    q_rot = _apply_pairwise_rotary(q_pre, angles_cumsum)
    k_rot = _apply_pairwise_rotary(k_pre, angles_cumsum)

    dt = DT.float()
    trap = Trap.float().clamp(0.0, 1.0)

    shifted = torch.zeros_like(dt)
    if seqlen > 1:
        shifted[:, :, :-1] = dt[:, :, 1:] * (1.0 - trap[:, :, 1:])
    gamma = dt * trap
    scale = gamma + shifted

    if Input_States is None and not return_final_states:
        da_cs = torch.cumsum(ADT.float(), dim=-1)
        decay = torch.exp(da_cs[:, :, :, None] - da_cs[:, :, None, :])
        positions = torch.arange(seqlen, device=Q.device)
        strictly_causal = positions[:, None] > positions[None, :]
        weights = torch.where(
            strictly_causal[None, None],
            decay * scale[:, :, None, :],
            torch.zeros((), device=Q.device, dtype=decay.dtype),
        )
        weights = weights + torch.diag_embed(gamma)

        qk_scores = torch.einsum("bthd,bshd->bhts", q_rot.float(), k_rot.float())
        out_f = torch.einsum("bhts,bshp->bthp", qk_scores * weights, V.float())
        if D is not None:
            out_f = out_f + D.float()[None, None, :, None] * V.float()
        if Z is not None:
            out_f = out_f * F.silu(Z.float())
        return out_f.to(V.dtype)

    alpha = torch.exp(ADT.float())
    ssm_state = input_ssm_state.float()
    if Input_States is not None:
        input_kv = input_v_prev1.float().unsqueeze(-1) * input_k_prev1.float().unsqueeze(-2)
        input_scale = dt[:, :, 0] * (1.0 - trap[:, :, 0])
        ssm_state = ssm_state + input_kv * input_scale[:, :, None, None]

    out = torch.empty((batch, seqlen, nheads, headdim_v), device=V.device, dtype=V.dtype)
    d_term = D.float()[None, :, None] if D is not None else None

    for t in range(seqlen):
        v_t = V[:, t].float()
        q_t = q_rot[:, t].float()
        k_t = k_rot[:, t].float()

        decayed_state = alpha[:, :, t, None, None] * ssm_state
        y_t = torch.einsum("bhpn,bhn->bhp", decayed_state, q_t)
        qk_dot = torch.sum(q_t * k_t, dim=-1)
        y_t = y_t + (gamma[:, :, t] * qk_dot)[:, :, None] * v_t
        if d_term is not None:
            y_t = y_t + d_term * v_t
        if Z is not None:
            y_t = y_t * F.silu(Z[:, t].float())
        out[:, t] = y_t.to(V.dtype)

        kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        ssm_state = decayed_state + scale[:, :, t, None, None] * kv_t

    if return_final_states:
        final_k_prev1 = k_rot[:, -1].to(K.dtype)
        final_v_prev1 = V[:, -1].to(V.dtype)
        final_k_prev2 = torch.zeros_like(final_k_prev1)
        final_v_prev2 = torch.zeros_like(final_v_prev1)
        return (
            out,
            final_angle_state,
            ssm_state,
            final_k_prev1,
            final_k_prev2,
            final_v_prev1,
            final_v_prev2,
        )
    return out


def simamba_siso_step(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Simpson: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    Input_States: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    Midpoint: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    boundary_mode: str = SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Single-token Simamba step.

    Shapes:
        Q/K: (batch, nheads, headdim_qk)
        V: (batch, nheads, headdim_v)
        ADT/DT/Simpson: (batch, nheads)
        Angles: (batch, nheads, n_angles)
    """
    _validate_boundary_mode(boundary_mode)

    batch, nheads_qk, headdim_qk = Q.shape
    if K.shape != Q.shape:
        raise ValueError(f"Q and K shape mismatch: {Q.shape} vs {K.shape}.")
    _, nheads, headdim_v = V.shape
    if nheads % nheads_qk != 0:
        raise ValueError(
            f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})."
        )
    if ADT.shape != (batch, nheads):
        raise ValueError(f"ADT shape mismatch: got {ADT.shape}.")
    if DT.shape != (batch, nheads):
        raise ValueError(f"DT shape mismatch: got {DT.shape}.")
    if Simpson.shape != (batch, nheads):
        raise ValueError(f"Simpson shape mismatch: got {Simpson.shape}.")
    if Midpoint is not None and Midpoint.shape != (batch, nheads):
        raise ValueError(f"Midpoint shape mismatch: got {Midpoint.shape}.")
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"Q_bias shape mismatch: got {Q_bias.shape}.")
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(f"K_bias shape mismatch: got {K_bias.shape}.")

    if nheads_qk != nheads:
        gqa_ratio = nheads // nheads_qk
        Q = Q.repeat_interleave(gqa_ratio, dim=1)
        K = K.repeat_interleave(gqa_ratio, dim=1)

    n_angles = Angles.shape[-1]
    if Angles.shape != (batch, nheads, n_angles):
        raise ValueError(f"Angles shape mismatch: got {Angles.shape}.")

    angle_state, ssm_state, k_prev1, k_prev2, v_prev1, v_prev2 = Input_States
    if angle_state.shape != (batch, nheads, n_angles):
        raise ValueError(f"Angle state shape mismatch: got {angle_state.shape}.")
    if ssm_state.shape != (batch, nheads, headdim_v, headdim_qk):
        raise ValueError(f"SSM state shape mismatch: got {ssm_state.shape}.")
    if k_prev1.shape != (batch, nheads, headdim_qk):
        raise ValueError(f"K prev1 shape mismatch: got {k_prev1.shape}.")
    if k_prev2.shape != (batch, nheads, headdim_qk):
        raise ValueError(f"K prev2 shape mismatch: got {k_prev2.shape}.")
    if v_prev1.shape != (batch, nheads, headdim_v):
        raise ValueError(f"V prev1 shape mismatch: got {v_prev1.shape}.")
    if v_prev2.shape != (batch, nheads, headdim_v):
        raise ValueError(f"V prev2 shape mismatch: got {v_prev2.shape}.")

    angle_inc = torch.tanh(Angles.float()) * DT.float().unsqueeze(-1) * math.pi
    next_angle_state = torch.remainder(angle_state.float() + angle_inc, 2.0 * math.pi)

    q_pre = Q + Q_bias[None, :, :]
    k_pre = K + K_bias[None, :, :]
    q_rot = _apply_pairwise_rotary(q_pre, next_angle_state)
    k_rot = _apply_pairwise_rotary(k_pre, next_angle_state)

    ssm = ssm_state.float()
    v_t = V.float()
    q_t = q_rot.float()
    k_t = k_rot.float()

    kv_t = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
    kv_prev1 = v_prev1.float().unsqueeze(-1) * k_prev1.float().unsqueeze(-2)
    kv_prev2 = v_prev2.float().unsqueeze(-1) * k_prev2.float().unsqueeze(-2)

    ssm = _simpson_state_update(ssm, kv_t, kv_prev1, kv_prev2, ADT, DT, Simpson, Midpoint)

    y = torch.einsum("bhpn,bhn->bhp", ssm, q_t)
    if D is not None:
        y = y + D.float()[None, :, None] * v_t
    if Z is not None:
        y = y * F.silu(Z.float())

    output_states = (
        next_angle_state,
        ssm,
        k_t.to(K.dtype),
        k_prev1.to(K.dtype),
        v_t.to(V.dtype),
        v_prev1.to(V.dtype),
    )
    return y.to(V.dtype), output_states
