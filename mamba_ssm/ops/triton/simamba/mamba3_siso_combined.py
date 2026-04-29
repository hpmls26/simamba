"""Simamba Triton autograd wrapper.

This keeps the Triton forward path but supplies a backward pass that respects
Simamba's Simpson discretization instead of reusing the trapezoidal Mamba-3
coefficient math.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
import triton

from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo
from mamba_ssm.ops.triton.simamba.mamba3_siso_bwd import (
    _compute_chunk_start_states,
    compute_native_simamba_grads,
    compute_dcoeffs,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_fwd import mamba3_siso_fwd
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import simamba_siso_combined


def _triton_alloc_fn(size: int, alignment: int, stream: Optional[int]):
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


try:
    triton.set_allocator(_triton_alloc_fn)
except Exception:
    pass


def _clone_for_grad(tensor: Optional[Tensor], requires_grad: bool) -> Optional[Tensor]:
    if tensor is None:
        return None
    clone = tensor.detach()
    if requires_grad:
        clone = clone.requires_grad_(True)
    return clone


def _zero_if_none(grad: Optional[Tensor], like: Tensor) -> Tensor:
    return torch.zeros_like(like) if grad is None else grad


_SEQUENCE_INPUT_NAMES = {
    "Q",
    "K",
    "V",
    "ADT",
    "DT",
    "Simpson",
    "Midpoint",
    "Angles",
    "Z",
}
_STATE_INPUT_NAMES = (
    "Input_Angle_State",
    "Input_SSM_State",
    "Input_K_Prev1_State",
    "Input_K_Prev2_State",
    "Input_V_Prev1_State",
    "Input_V_Prev2_State",
)


def _slice_sequence_tensor(name: str, tensor: Optional[Tensor], start: int, end: int) -> Optional[Tensor]:
    if tensor is None:
        return None
    if name in {"ADT", "DT", "Simpson", "Midpoint"}:
        return tensor[:, :, start:end]
    return tensor[:, start:end]


def _reference_autograd_grads(
    *,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Simpson: Tensor,
    Midpoint: Optional[Tensor],
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    D: Optional[Tensor],
    Z: Optional[Tensor],
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    grad_out: Optional[Tensor],
    grad_final_angle_state: Optional[Tensor],
    grad_final_ssm_state: Optional[Tensor],
    grad_final_k_prev1_state: Optional[Tensor],
    grad_final_k_prev2_state: Optional[Tensor],
    grad_final_v_prev1_state: Optional[Tensor],
    grad_final_v_prev2_state: Optional[Tensor],
    return_final_states: bool,
    needs_grad: dict[str, bool],
    recompute_chunk_size: int,
) -> dict[str, Optional[Tensor]]:
    requested_input_names = [name for name, needed in needs_grad.items() if needed and name not in _STATE_INPUT_NAMES]
    wants_input_state_grads = any(needs_grad.get(name, False) for name in _STATE_INPUT_NAMES)
    if not requested_input_names and not wants_input_state_grads:
        return {}

    seqlen = Q.shape[1]
    chunk_size = max(1, min(recompute_chunk_size, seqlen))
    chunk_start_states, _ = _compute_chunk_start_states(
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

    accumulated_grads: dict[str, Optional[Tensor]] = {name: None for name in requested_input_names}
    next_state_grads: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = None

    for chunk_idx in range(len(chunk_start_states) - 1, -1, -1):
        start = chunk_idx * chunk_size
        end = min(seqlen, start + chunk_size)

        local_requested_names = []
        local_requested_tensors = []

        def prep_local(name: str, tensor: Optional[Tensor]) -> Optional[Tensor]:
            requires_grad = needs_grad.get(name, False)
            ref = _clone_for_grad(tensor, requires_grad)
            if requires_grad:
                local_requested_names.append(name)
                local_requested_tensors.append(ref)
            return ref

        q_ref = prep_local("Q", _slice_sequence_tensor("Q", Q, start, end))
        k_ref = prep_local("K", _slice_sequence_tensor("K", K, start, end))
        v_ref = prep_local("V", _slice_sequence_tensor("V", V, start, end))
        adt_ref = prep_local("ADT", _slice_sequence_tensor("ADT", ADT, start, end))
        dt_ref = prep_local("DT", _slice_sequence_tensor("DT", DT, start, end))
        simpson_ref = prep_local("Simpson", _slice_sequence_tensor("Simpson", Simpson, start, end))
        midpoint_ref = prep_local("Midpoint", _slice_sequence_tensor("Midpoint", Midpoint, start, end))
        q_bias_ref = prep_local("Q_bias", Q_bias)
        k_bias_ref = prep_local("K_bias", K_bias)
        angles_ref = prep_local("Angles", _slice_sequence_tensor("Angles", Angles, start, end))
        d_ref = prep_local("D", D)
        z_ref = prep_local("Z", _slice_sequence_tensor("Z", Z, start, end))

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

        with torch.enable_grad():
            outputs_ref = simamba_siso_combined(
                Q=q_ref if q_ref is not None else _slice_sequence_tensor("Q", Q, start, end).detach(),
                K=k_ref if k_ref is not None else _slice_sequence_tensor("K", K, start, end).detach(),
                V=v_ref if v_ref is not None else _slice_sequence_tensor("V", V, start, end).detach(),
                ADT=adt_ref if adt_ref is not None else _slice_sequence_tensor("ADT", ADT, start, end).detach(),
                DT=dt_ref if dt_ref is not None else _slice_sequence_tensor("DT", DT, start, end).detach(),
                Simpson=simpson_ref if simpson_ref is not None else _slice_sequence_tensor("Simpson", Simpson, start, end).detach(),
                Midpoint=midpoint_ref if midpoint_ref is not None else (_slice_sequence_tensor("Midpoint", Midpoint, start, end).detach() if Midpoint is not None else None),
                Q_bias=q_bias_ref if q_bias_ref is not None else Q_bias.detach(),
                K_bias=k_bias_ref if k_bias_ref is not None else K_bias.detach(),
                Angles=angles_ref if angles_ref is not None else _slice_sequence_tensor("Angles", Angles, start, end).detach(),
                D=d_ref if d_ref is not None else (D.detach() if D is not None else None),
                Z=z_ref if z_ref is not None else (_slice_sequence_tensor("Z", Z, start, end).detach() if Z is not None else None),
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
            if return_final_states and chunk_idx == len(chunk_start_states) - 1:
                next_state_grads = (
                    _zero_if_none(grad_final_angle_state, final_angle_ref),
                    _zero_if_none(grad_final_ssm_state, final_ssm_ref),
                    _zero_if_none(grad_final_k_prev1_state, final_k_prev1_ref),
                    _zero_if_none(grad_final_k_prev2_state, final_k_prev2_ref),
                    _zero_if_none(grad_final_v_prev1_state, final_v_prev1_ref),
                    _zero_if_none(grad_final_v_prev2_state, final_v_prev2_ref),
                )
            else:
                next_state_grads = (
                    torch.zeros_like(final_angle_ref),
                    torch.zeros_like(final_ssm_ref),
                    torch.zeros_like(final_k_prev1_ref),
                    torch.zeros_like(final_k_prev2_ref),
                    torch.zeros_like(final_v_prev1_ref),
                    torch.zeros_like(final_v_prev2_ref),
                )

        grads = torch.autograd.grad(
            outputs=outputs_ref,
            inputs=(*local_requested_tensors, *input_states_ref),
            grad_outputs=(
                _zero_if_none(grad_out, out_ref)[:, start:end] if grad_out is not None else torch.zeros_like(out_ref),
                *next_state_grads,
            ),
            allow_unused=False,
        )

        local_input_grads = grads[: len(local_requested_tensors)]
        next_state_grads = grads[len(local_requested_tensors) :]

        for name, grad in zip(local_requested_names, local_input_grads):
            current = accumulated_grads[name]
            if name in _SEQUENCE_INPUT_NAMES:
                if current is None:
                    current = torch.zeros_like(
                        {"Q": Q, "K": K, "V": V, "ADT": ADT, "DT": DT, "Simpson": Simpson, "Midpoint": Midpoint, "Angles": Angles, "Z": Z}[name]
                    )
                if name in {"ADT", "DT", "Simpson", "Midpoint"}:
                    current[:, :, start:end] = current[:, :, start:end] + grad
                else:
                    current[:, start:end] = current[:, start:end] + grad
                accumulated_grads[name] = current
            else:
                accumulated_grads[name] = grad if current is None else current + grad

    if wants_input_state_grads and next_state_grads is not None:
        for name, grad in zip(_STATE_INPUT_NAMES, next_state_grads):
            if needs_grad.get(name, False):
                accumulated_grads[name] = grad

    return accumulated_grads


class _SimambaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        ADT: Tensor,
        DT: Tensor,
        Simpson: Tensor,
        Midpoint: Optional[Tensor],
        Q_bias: Tensor,
        K_bias: Tensor,
        Angles: Tensor,
        D: Optional[Tensor],
        Z: Optional[Tensor],
        Input_Angle_State: Optional[Tensor],
        Input_SSM_State: Optional[Tensor],
        Input_K_Prev1_State: Optional[Tensor],
        Input_K_Prev2_State: Optional[Tensor],
        Input_V_Prev1_State: Optional[Tensor],
        Input_V_Prev2_State: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        chunk_size: int,
        recompute_chunk_size: int,
        return_final_states: bool,
    ):
        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass

        Input_States = None
        if Input_SSM_State is not None:
            Input_States = (
                Input_Angle_State,
                Input_SSM_State,
                Input_K_Prev1_State,
                Input_K_Prev2_State,
                Input_V_Prev1_State,
                Input_V_Prev2_State,
            )

        out, *_, final_states = mamba3_siso_fwd(
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
            Initial_States=Input_States,
            chunk_size=chunk_size,
            store_states_adt_outv=False,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )

        ctx.chunk_size = chunk_size
        ctx.recompute_chunk_size = recompute_chunk_size
        ctx.return_final_states = return_final_states
        ctx.has_midpoint = Midpoint is not None
        ctx.has_D = D is not None
        ctx.has_Z = Z is not None
        ctx.has_input_states = Input_States is not None
        ctx.has_varlen = cu_seqlens is not None

        if any(ctx.needs_input_grad):
            midpoint_save = Midpoint if Midpoint is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            d_save = D if D is not None else torch.empty((), device=Q.device, dtype=torch.float32)
            z_save = Z if Z is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            angle_state_save = (
                Input_Angle_State if Input_Angle_State is not None else torch.empty((), device=Q.device, dtype=torch.float32)
            )
            ssm_state_save = (
                Input_SSM_State if Input_SSM_State is not None else torch.empty((), device=Q.device, dtype=torch.float32)
            )
            k_prev1_save = (
                Input_K_Prev1_State if Input_K_Prev1_State is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            )
            k_prev2_save = (
                Input_K_Prev2_State if Input_K_Prev2_State is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            )
            v_prev1_save = (
                Input_V_Prev1_State if Input_V_Prev1_State is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            )
            v_prev2_save = (
                Input_V_Prev2_State if Input_V_Prev2_State is not None else torch.empty((), device=Q.device, dtype=Q.dtype)
            )
            cu_seqlens_save = (
                cu_seqlens if cu_seqlens is not None else torch.empty((), device=Q.device, dtype=torch.int32)
            )
            ctx.save_for_backward(
                Q,
                K,
                V,
                ADT,
                DT,
                Simpson,
                midpoint_save,
                Q_bias,
                K_bias,
                Angles,
                d_save,
                z_save,
                angle_state_save,
                ssm_state_save,
                k_prev1_save,
                k_prev2_save,
                v_prev1_save,
                v_prev2_save,
                cu_seqlens_save,
            )
        else:
            ctx.save_for_backward()

        if return_final_states:
            (
                final_angle_state,
                final_ssm_state,
                final_k_prev1_state,
                final_k_prev2_state,
                final_v_prev1_state,
                final_v_prev2_state,
            ) = final_states
            return (
                out,
                final_angle_state,
                final_ssm_state,
                final_k_prev1_state,
                final_k_prev2_state,
                final_v_prev1_state,
                final_v_prev2_state,
            )
        return out

    @staticmethod
    def backward(
        ctx,
        grad_out: Optional[Tensor] = None,
        grad_final_angle_state: Optional[Tensor] = None,
        grad_final_ssm_state: Optional[Tensor] = None,
        grad_final_k_prev1_state: Optional[Tensor] = None,
        grad_final_k_prev2_state: Optional[Tensor] = None,
        grad_final_v_prev1_state: Optional[Tensor] = None,
        grad_final_v_prev2_state: Optional[Tensor] = None,
    ):
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError("Backward called without saved tensors.")
        if (
            grad_out is None
            and grad_final_angle_state is None
            and grad_final_ssm_state is None
            and grad_final_k_prev1_state is None
            and grad_final_k_prev2_state is None
            and grad_final_v_prev1_state is None
            and grad_final_v_prev2_state is None
        ):
            raise RuntimeError("No gradients provided for Simamba backward.")

        if ctx.has_varlen:
            raise NotImplementedError("Simamba Triton autograd does not support cu_seqlens yet.")

        (
            Q,
            K,
            V,
            ADT,
            DT,
            Simpson,
            midpoint_save,
            Q_bias,
            K_bias,
            Angles,
            d_save,
            z_save,
            angle_state_save,
            ssm_state_save,
            k_prev1_save,
            k_prev2_save,
            v_prev1_save,
            v_prev2_save,
            _,
        ) = ctx.saved_tensors

        Midpoint = midpoint_save if ctx.has_midpoint else None
        D = d_save if ctx.has_D else None
        Z = z_save if ctx.has_Z else None
        Input_States = None
        if ctx.has_input_states:
            Input_States = (
                angle_state_save,
                ssm_state_save,
                k_prev1_save,
                k_prev2_save,
                v_prev1_save,
                v_prev2_save,
            )

        needs = ctx.needs_input_grad
        grad_out_native = _zero_if_none(grad_out, torch.zeros_like(V))

        with torch.no_grad():
            (
                out_native,
                out_pregate_native,
                angles_cumsum,
                chunk_ssm_starts,
                chunk_k_prev1_starts,
                chunk_k_prev2_starts,
                chunk_v_prev1_starts,
                chunk_v_prev2_starts,
                final_states_native,
            ) = mamba3_siso_fwd(
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
                Initial_States=Input_States,
                chunk_size=ctx.chunk_size,
                store_states_adt_outv=True,
                return_final_states=True,
                cu_seqlens=None,
            )

        if (
            angles_cumsum is not None
            and chunk_ssm_starts is not None
            and chunk_k_prev1_starts is not None
            and chunk_k_prev2_starts is not None
            and chunk_v_prev1_starts is not None
            and chunk_v_prev2_starts is not None
            and final_states_native is not None
        ):
            _, final_ssm_state_native, _, _, _, _ = final_states_native
            if Z is not None:
                dZ_native, grad_out_native = compute_dzdo(
                    grad_out_native,
                    Z,
                    out_pregate_native,
                    chunk_size=ctx.chunk_size,
                )
            else:
                dZ_native = None

            (
                dQ_native,
                dK_native,
                dV_native,
                dADT_native,
                dDT_native,
                dSimpson_native,
                dMidpoint_native,
                dAngles_cumsum_native,
                dQ_bias_native,
                dK_bias_native,
                dD_native,
                dInput_SSM_native,
                dInput_K_Prev1_native,
                dInput_K_Prev2_native,
                dInput_V_Prev1_native,
                dInput_V_Prev2_native,
            ) = compute_native_simamba_grads(
                Q=Q,
                K=K,
                V=V,
                ADT=ADT,
                DT=DT,
                Simpson=Simpson,
                Midpoint=Midpoint,
                Q_bias=Q_bias,
                K_bias=K_bias,
                Angles_Cumsum=angles_cumsum,
                D=D,
                grad_out=grad_out_native,
                chunk_start_ssm_state=chunk_ssm_starts,
                chunk_start_k_prev1_state=chunk_k_prev1_starts,
                chunk_start_k_prev2_state=chunk_k_prev2_starts,
                chunk_start_v_prev1_state=chunk_v_prev1_starts,
                chunk_start_v_prev2_state=chunk_v_prev2_starts,
                final_ssm_state=final_ssm_state_native,
                grad_final_ssm_state=grad_final_ssm_state,
                grad_final_k_prev1_state=grad_final_k_prev1_state,
                grad_final_k_prev2_state=grad_final_k_prev2_state,
                grad_final_v_prev1_state=grad_final_v_prev1_state,
                grad_final_v_prev2_state=grad_final_v_prev2_state,
                chunk_size=ctx.chunk_size,
            )

            dAngles_native, dDT_angle_native, dInput_Angle_native = angle_dt_bwd(
                grad_out=dAngles_cumsum_native,
                angle=Angles,
                dt=DT,
                has_init_state=ctx.has_input_states,
                chunk_size=ctx.chunk_size,
                grad_output_state=grad_final_angle_state if ctx.return_final_states else None,
                cu_seqlens=None,
            )

            dDT_native = dDT_native + dDT_angle_native

            return (
                dQ_native if needs[0] else None,
                dK_native if needs[1] else None,
                dV_native if needs[2] else None,
                dADT_native if needs[3] else None,
                dDT_native if needs[4] else None,
                dSimpson_native if needs[5] else None,
                dMidpoint_native if (needs[6] and ctx.has_midpoint) else None,
                dQ_bias_native if needs[7] else None,
                dK_bias_native if needs[8] else None,
                dAngles_native if needs[9] else None,
                dD_native if (needs[10] and ctx.has_D) else None,
                dZ_native if (needs[11] and ctx.has_Z) else None,
                dInput_Angle_native if (needs[12] and ctx.has_input_states) else None,
                dInput_SSM_native if (needs[13] and ctx.has_input_states) else None,
                dInput_K_Prev1_native.to(k_prev1_save.dtype) if (needs[14] and ctx.has_input_states) else None,
                dInput_K_Prev2_native.to(k_prev2_save.dtype) if (needs[15] and ctx.has_input_states) else None,
                dInput_V_Prev1_native.to(v_prev1_save.dtype) if (needs[16] and ctx.has_input_states) else None,
                dInput_V_Prev2_native.to(v_prev2_save.dtype) if (needs[17] and ctx.has_input_states) else None,
                None,
                None,
                None,
                None,
            )

        full_reference = ctx.return_final_states or ctx.has_input_states

        if full_reference:
            ref_grads = _reference_autograd_grads(
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
                Input_States=Input_States,
                grad_out=grad_out,
                grad_final_angle_state=grad_final_angle_state,
                grad_final_ssm_state=grad_final_ssm_state,
                grad_final_k_prev1_state=grad_final_k_prev1_state,
                grad_final_k_prev2_state=grad_final_k_prev2_state,
                grad_final_v_prev1_state=grad_final_v_prev1_state,
                grad_final_v_prev2_state=grad_final_v_prev2_state,
                return_final_states=ctx.return_final_states,
                needs_grad={
                    "Q": needs[0],
                    "K": needs[1],
                    "V": needs[2],
                    "ADT": needs[3],
                    "DT": needs[4],
                    "Simpson": needs[5],
                    "Midpoint": needs[6] and ctx.has_midpoint,
                    "Q_bias": needs[7],
                    "K_bias": needs[8],
                    "Angles": needs[9],
                    "D": needs[10] and ctx.has_D,
                    "Z": needs[11] and ctx.has_Z,
                    "Input_Angle_State": needs[12] and ctx.has_input_states,
                    "Input_SSM_State": needs[13] and ctx.has_input_states,
                    "Input_K_Prev1_State": needs[14] and ctx.has_input_states,
                    "Input_K_Prev2_State": needs[15] and ctx.has_input_states,
                    "Input_V_Prev1_State": needs[16] and ctx.has_input_states,
                    "Input_V_Prev2_State": needs[17] and ctx.has_input_states,
                },
                recompute_chunk_size=ctx.recompute_chunk_size,
            )
            return (
                ref_grads.get("Q"),
                ref_grads.get("K"),
                ref_grads.get("V"),
                ref_grads.get("ADT"),
                ref_grads.get("DT"),
                ref_grads.get("Simpson"),
                ref_grads.get("Midpoint"),
                ref_grads.get("Q_bias"),
                ref_grads.get("K_bias"),
                ref_grads.get("Angles"),
                ref_grads.get("D"),
                ref_grads.get("Z"),
                ref_grads.get("Input_Angle_State"),
                ref_grads.get("Input_SSM_State"),
                ref_grads.get("Input_K_Prev1_State"),
                ref_grads.get("Input_K_Prev2_State"),
                ref_grads.get("Input_V_Prev1_State"),
                ref_grads.get("Input_V_Prev2_State"),
                None,
                None,
                None,
                None,
            )

        dADT = dDT = dSimpson = dMidpoint = None
        if needs[3] or needs[4] or needs[5] or (needs[6] and ctx.has_midpoint):
            with torch.enable_grad():
                dADT_ref, dDT_ref, dSimpson_ref, dMidpoint_ref = compute_dcoeffs(
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
                    grad_out=_zero_if_none(grad_out, torch.zeros_like(V)),
                    Input_States=None,
                    recompute_chunk_size=ctx.recompute_chunk_size,
                    compute_dt_grad=needs[4],
                )
            dADT = dADT_ref if needs[3] else None
            dDT = dDT_ref if needs[4] else None
            dSimpson = dSimpson_ref if needs[5] else None
            dMidpoint = dMidpoint_ref if (needs[6] and ctx.has_midpoint) else None

        ref_grads = _reference_autograd_grads(
            Q=Q,
            K=K,
            V=V,
            ADT=ADT.detach(),
            DT=DT.detach(),
            Simpson=Simpson.detach(),
            Midpoint=Midpoint.detach() if Midpoint is not None else None,
            Q_bias=Q_bias,
            K_bias=K_bias,
            Angles=Angles,
            D=D,
            Z=Z,
            Input_States=None,
            grad_out=grad_out,
            grad_final_angle_state=None,
            grad_final_ssm_state=None,
            grad_final_k_prev1_state=None,
            grad_final_k_prev2_state=None,
            grad_final_v_prev1_state=None,
            grad_final_v_prev2_state=None,
            return_final_states=False,
            needs_grad={
                "Q": needs[0],
                "K": needs[1],
                "V": needs[2],
                "Q_bias": needs[7],
                "K_bias": needs[8],
                "Angles": needs[9],
                "D": needs[10] and ctx.has_D,
                "Z": needs[11] and ctx.has_Z,
            },
            recompute_chunk_size=ctx.recompute_chunk_size,
        )

        return (
            ref_grads.get("Q"),
            ref_grads.get("K"),
            ref_grads.get("V"),
            dADT,
            dDT,
            dSimpson,
            dMidpoint,
            ref_grads.get("Q_bias"),
            ref_grads.get("K_bias"),
            ref_grads.get("Angles"),
            ref_grads.get("D"),
            ref_grads.get("Z"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mamba3_siso_combined(
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
    Initial_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = None,
    chunk_size: int = 64,
    recompute_chunk_size: Optional[int] = None,
    return_final_states: bool = False,
    cu_seqlens: Optional[Tensor] = None,
):
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, _ = V.shape
    if nheads % nheads_qk != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk}).")
    if headdim_qk % 2 != 0:
        raise ValueError(f"headdim_qk ({headdim_qk}) must be even for rotary embeddings.")
    if cu_seqlens is not None and batch != 1:
        raise ValueError(f"Batch size must be 1 with cu_seqlens, got batch={batch}.")

    (
        Input_Angle_State,
        Input_SSM_State,
        Input_K_Prev1_State,
        Input_K_Prev2_State,
        Input_V_Prev1_State,
        Input_V_Prev2_State,
    ) = Initial_States if Initial_States is not None else (None, None, None, None, None, None)

    Q = Q.to(torch.bfloat16)
    K = K.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    Simpson = Simpson.to(torch.bfloat16)
    Midpoint = Midpoint.to(torch.bfloat16) if Midpoint is not None else None
    Angles = Angles.to(torch.bfloat16)
    if Z is not None:
        Z = Z.to(torch.bfloat16)

    if recompute_chunk_size is None:
        recompute_chunk_size = chunk_size

    return _SimambaFunction.apply(
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
        Input_K_Prev1_State,
        Input_K_Prev2_State,
        Input_V_Prev1_State,
        Input_V_Prev2_State,
        cu_seqlens,
        chunk_size,
        recompute_chunk_size,
        return_final_states,
    )
