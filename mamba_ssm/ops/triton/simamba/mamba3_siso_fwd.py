"""Simamba SISO forward wrapper built on the Triton step kernel.

Phase 5 focuses on locking forward semantics against the reference equations
before introducing fused chunk kernels.
"""

from typing import Optional, Tuple

import torch

from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step


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
    """Simamba forward path using repeated Triton step calls.

    Returns an API-compatible tuple with intermediate slots set to ``None`` for
    now. This keeps the function shape aligned with the original Mamba-3
    wrapper while we iterate on fused kernels.
    """
    del chunk_size, store_states_adt_outv

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    if K.shape != Q.shape:
        raise ValueError(f"Q and K shape mismatch: {Q.shape} vs {K.shape}.")
    _, _, nheads, headdim_v = V.shape

    if nheads % nheads_qk != 0:
        raise ValueError(
            f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})."
        )
    if ADT.shape != (batch, nheads, seqlen):
        raise ValueError(f"ADT shape mismatch: expected {(batch, nheads, seqlen)}, got {ADT.shape}.")
    if DT.shape != (batch, nheads, seqlen):
        raise ValueError(f"DT shape mismatch: expected {(batch, nheads, seqlen)}, got {DT.shape}.")
    if Simpson.shape != (batch, nheads, seqlen):
        raise ValueError(
            f"Simpson shape mismatch: expected {(batch, nheads, seqlen)}, got {Simpson.shape}."
        )
    if Midpoint is not None and Midpoint.shape != (batch, nheads, seqlen):
        raise ValueError(
            f"Midpoint shape mismatch: expected {(batch, nheads, seqlen)}, got {Midpoint.shape}."
        )
    if Q_bias.shape != (nheads, headdim_qk):
        raise ValueError(
            f"Q_bias shape mismatch: expected {(nheads, headdim_qk)}, got {Q_bias.shape}."
        )
    if K_bias.shape != (nheads, headdim_qk):
        raise ValueError(
            f"K_bias shape mismatch: expected {(nheads, headdim_qk)}, got {K_bias.shape}."
        )
    if D is not None and D.shape != (nheads,):
        raise ValueError(f"D shape mismatch: expected {(nheads,)}, got {D.shape}.")
    if Z is not None and Z.shape != (batch, seqlen, nheads, headdim_v):
        raise ValueError(
            f"Z shape mismatch: expected {(batch, seqlen, nheads, headdim_v)}, got {Z.shape}."
        )

    n_angles = Angles.shape[-1]
    if Angles.shape != (batch, seqlen, nheads, n_angles):
        raise ValueError(
            f"Angles shape mismatch: expected {(batch, seqlen, nheads, n_angles)}, got {Angles.shape}."
        )

    is_varlen = cu_seqlens is not None
    if is_varlen:
        if batch != 1:
            raise ValueError(
                f"Varlen mode requires batch=1, got batch={batch}."
            )
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
        if init_angle_state.shape[0] != state_batch:
            raise ValueError(
                f"Initial angle state batch mismatch: expected {state_batch}, got {init_angle_state.shape[0]}."
            )
        if init_ssm_state.shape[0] != state_batch:
            raise ValueError(
                f"Initial SSM state batch mismatch: expected {state_batch}, got {init_ssm_state.shape[0]}."
            )

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
        for t in range(seqlen):
            out_t, states = mamba3_siso_step(
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
                Input_States=states,
            )
            out[:, t] = out_t

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

            for t in range(start, end):
                out_t, states = mamba3_siso_step(
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
                    Input_States=states,
                )
                out[:, t] = out_t

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

    return (
        out,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        final_states,
    )