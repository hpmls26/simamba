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

from mamba_ssm.ops.triton.simamba.mamba3_siso_bwd import compute_dcoeffs
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
) -> dict[str, Optional[Tensor]]:
    requested_names = []
    requested_tensors = []

    def prep(name: str, tensor: Optional[Tensor]) -> Optional[Tensor]:
        requires_grad = needs_grad.get(name, False)
        ref = _clone_for_grad(tensor, requires_grad)
        if requires_grad:
            requested_names.append(name)
            requested_tensors.append(ref)
        return ref

    q_ref = prep("Q", Q)
    k_ref = prep("K", K)
    v_ref = prep("V", V)
    adt_ref = prep("ADT", ADT)
    dt_ref = prep("DT", DT)
    simpson_ref = prep("Simpson", Simpson)
    midpoint_ref = prep("Midpoint", Midpoint)
    q_bias_ref = prep("Q_bias", Q_bias)
    k_bias_ref = prep("K_bias", K_bias)
    angles_ref = prep("Angles", Angles)
    d_ref = prep("D", D)
    z_ref = prep("Z", Z)

    input_states_ref = None
    if Input_States is not None:
        state_names = (
            "Input_Angle_State",
            "Input_SSM_State",
            "Input_K_Prev1_State",
            "Input_K_Prev2_State",
            "Input_V_Prev1_State",
            "Input_V_Prev2_State",
        )
        input_states_ref = tuple(prep(name, state) for name, state in zip(state_names, Input_States))

    if not requested_tensors:
        return {}

    with torch.enable_grad():
        outputs_ref = simamba_siso_combined(
            Q=q_ref,
            K=k_ref,
            V=v_ref,
            ADT=adt_ref,
            DT=dt_ref,
            Simpson=simpson_ref,
            Midpoint=midpoint_ref,
            Q_bias=q_bias_ref,
            K_bias=k_bias_ref,
            Angles=angles_ref,
            D=d_ref,
            Z=z_ref,
            Input_States=input_states_ref,
            return_final_states=return_final_states,
        )

    if return_final_states:
        (
            out_ref,
            final_angle_ref,
            final_ssm_ref,
            final_k_prev1_ref,
            final_k_prev2_ref,
            final_v_prev1_ref,
            final_v_prev2_ref,
        ) = outputs_ref
        outputs = outputs_ref
        grad_outputs = (
            _zero_if_none(grad_out, out_ref),
            _zero_if_none(grad_final_angle_state, final_angle_ref),
            _zero_if_none(grad_final_ssm_state, final_ssm_ref),
            _zero_if_none(grad_final_k_prev1_state, final_k_prev1_ref),
            _zero_if_none(grad_final_k_prev2_state, final_k_prev2_ref),
            _zero_if_none(grad_final_v_prev1_state, final_v_prev1_ref),
            _zero_if_none(grad_final_v_prev2_state, final_v_prev2_ref),
        )
    else:
        outputs = (outputs_ref,)
        grad_outputs = (_zero_if_none(grad_out, outputs_ref),)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=requested_tensors,
        grad_outputs=grad_outputs,
        allow_unused=False,
    )
    return {name: grad for name, grad in zip(requested_names, grads)}


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
        return_final_states,
    )
