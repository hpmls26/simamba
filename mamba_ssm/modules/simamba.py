# Copyright (c) 2026, Dao AI Lab, Goombalab.

import math

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (
    SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
    SIMAMBA_SUPPORTED_BOUNDARY_MODES,
    simamba_siso_combined,
    simamba_siso_step,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_fwd import mamba3_siso_fwd as simamba_triton_siso_fwd
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step as simamba_triton_siso_step


SIMAMBA_BACKEND_REFERENCE = "reference"
SIMAMBA_BACKEND_TRITON = "triton"
SIMAMBA_SUPPORTED_BACKENDS = (SIMAMBA_BACKEND_REFERENCE, SIMAMBA_BACKEND_TRITON)


class Simamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
        use_midpoint_control=False,
        simamba_backend=SIMAMBA_BACKEND_REFERENCE,
        simpson_boundary_mode=SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
        dropout=0.0,
        layer_idx=None,
        n_layer=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        del dropout, n_layer, kwargs, mimo_rank

        if is_mimo:
            raise NotImplementedError("Phase A Simamba currently supports SISO only.")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = False
        self.mimo_rank = 1
        self.use_midpoint_control = use_midpoint_control
        if simamba_backend not in SIMAMBA_SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported simamba_backend={simamba_backend!r}. "
                f"Expected one of {SIMAMBA_SUPPORTED_BACKENDS}."
            )
        self.simamba_backend = simamba_backend
        if simpson_boundary_mode not in SIMAMBA_SUPPORTED_BOUNDARY_MODES:
            raise ValueError(
                f"Unsupported simpson_boundary_mode={simpson_boundary_mode!r}. "
                f"Expected one of {SIMAMBA_SUPPORTED_BOUNDARY_MODES}."
            )
        self.simpson_boundary_mode = simpson_boundary_mode

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, simpson, (optional midpoint), angle]
        num_head_controls = 4 if self.use_midpoint_control else 3
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + num_head_controls * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )

        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        del seq_idx
        if cu_seqlens is not None and self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
            raise NotImplementedError("Reference Simamba backend does not support varlen yet.")
        if u.dim() != 3:
            raise ValueError(f"Expected input shape (batch, seqlen, d_model), got {u.shape}.")

        batch, seqlen, _ = u.shape

        angle_dt_state = None
        ssm_state = None
        k_prev1_state = None
        k_prev2_state = None
        v_prev1_state = None
        v_prev2_state = None

        if inference_params is not None:
            (
                angle_dt_state,
                ssm_state,
                k_prev1_state,
                k_prev2_state,
                v_prev1_state,
                v_prev2_state,
            ) = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                if seqlen != 1:
                    raise ValueError(
                        "Incremental decoding path expects seqlen=1 for forward(..., inference_params=...)."
                    )
                out, *_ = self.step(
                    u[:, 0],
                    angle_dt_state,
                    ssm_state,
                    k_prev1_state,
                    k_prev2_state,
                    v_prev1_state,
                    v_prev2_state,
                )
                return out.unsqueeze(1)

        zxBCdtAs = self.in_proj(u)
        split_sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads,
            self.d_state * self.num_bc_heads,
            self.nheads,
            self.nheads,
            self.nheads,
        ]
        if self.use_midpoint_control:
            split_sizes.append(self.nheads)
        split_sizes.append(self.num_rope_angles)

        splits = torch.split(zxBCdtAs, split_sizes, dim=-1)
        if self.use_midpoint_control:
            z, x, B, C, dd_dt, dd_A, simpson, midpoint, angles = splits
        else:
            z, x, B, C, dd_dt, dd_A, simpson, angles = splits
            midpoint = None

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        simpson = rearrange(torch.sigmoid(simpson), "b l h -> b h l")
        midpoint = rearrange(torch.sigmoid(midpoint), "b l h -> b h l") if midpoint is not None else None

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1)

        B = self.B_norm(B)
        C = self.C_norm(C)

        input_states = None
        if ssm_state is not None:
            input_states = (
                angle_dt_state,
                ssm_state,
                k_prev1_state,
                k_prev2_state,
                v_prev1_state,
                v_prev2_state,
            )

        if self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
            y = simamba_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=input_states,
                return_final_states=input_states is not None,
                boundary_mode=self.simpson_boundary_mode,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k1, last_k2, last_v1, last_v2 = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_prev1_state.copy_(last_k1)
                k_prev2_state.copy_(last_k2)
                v_prev1_state.copy_(last_v1)
                v_prev2_state.copy_(last_v2)
        else:
            y_tuple = simamba_triton_siso_fwd(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Initial_States=input_states,
                return_final_states=input_states is not None,
                cu_seqlens=cu_seqlens,
            )
            y = y_tuple[0]
            if ssm_state is not None:
                final_states = y_tuple[-1]
                if final_states is None:
                    raise RuntimeError("Triton Simamba path did not return final states.")
                last_angle, last_state, last_k1, last_k2, last_v1, last_v2 = final_states
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_prev1_state.copy_(last_k1)
                k_prev2_state.copy_(last_k2)
                v_prev1_state.copy_(last_v1)
                v_prev2_state.copy_(last_v2)

        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)

        out = self.out_proj(y.to(x.dtype))
        return out

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, simpson_proj, angle_proj, midpoint_proj=None):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        simpson = torch.sigmoid(simpson_proj)
        midpoint = torch.sigmoid(midpoint_proj) if midpoint_proj is not None else None

        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=1)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=1)

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1)
        C = C.expand(-1, -1, self.nheads, -1)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, simpson, midpoint, _A, angles

    def step(
        self,
        u,
        angle_state,
        ssm_state,
        k_prev1_state,
        k_prev2_state,
        v_prev1_state,
        v_prev2_state,
        **kwargs,
    ):
        del kwargs

        if u.dim() == 3:
            if u.shape[1] != 1:
                raise ValueError(f"Step expects seqlen=1 when input is 3D, got {u.shape}.")
            u = u[:, 0]
        if u.dim() != 2:
            raise ValueError(f"Step expects shape (batch, d_model), got {u.shape}.")

        zxBCdtAs = self.in_proj(u)
        split_sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads,
            self.d_state * self.num_bc_heads,
            self.nheads,
            self.nheads,
            self.nheads,
        ]
        if self.use_midpoint_control:
            split_sizes.append(self.nheads)
        split_sizes.append(self.num_rope_angles)

        splits = torch.split(zxBCdtAs, split_sizes, dim=-1)
        if self.use_midpoint_control:
            z, x, B, C, dd_dt, dd_A, simpson, midpoint, angles = splits
        else:
            z, x, B, C, dd_dt, dd_A, simpson, angles = splits
            midpoint = None

        DT, B, C, x, z, simpson, midpoint, A, angles = self._preprocess(
            dd_A,
            dd_dt,
            B,
            C,
            x,
            z,
            simpson,
            angles,
            midpoint,
        )

        if self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
            y, output_states = simamba_siso_step(
                Q=C.squeeze(1),
                K=B.squeeze(1),
                V=x,
                ADT=A * DT,
                DT=DT,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                Input_States=(
                    angle_state,
                    ssm_state,
                    k_prev1_state,
                    k_prev2_state,
                    v_prev1_state,
                    v_prev2_state,
                ),
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                boundary_mode=self.simpson_boundary_mode,
            )
        else:
            y, output_states = simamba_triton_siso_step(
                Q=C.squeeze(1),
                K=B.squeeze(1),
                V=x,
                ADT=A * DT,
                DT=DT,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                Input_States=(
                    angle_state,
                    ssm_state,
                    k_prev1_state,
                    k_prev2_state,
                    v_prev1_state,
                    v_prev2_state,
                ),
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
            )

        (
            nxt_angle_state,
            nxt_ssm_state,
            nxt_k_prev1_state,
            nxt_k_prev2_state,
            nxt_v_prev1_state,
            nxt_v_prev2_state,
        ) = output_states

        y = rearrange(y, "b h p -> b (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b h p -> b (h p)")
            y = self.norm(y, z)

        out = self.out_proj(y.to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        ssm_state.copy_(nxt_ssm_state)
        k_prev1_state.copy_(nxt_k_prev1_state)
        k_prev2_state.copy_(nxt_k_prev2_state)
        v_prev1_state.copy_(nxt_v_prev1_state)
        v_prev2_state.copy_(nxt_v_prev2_state)

        return (
            out,
            nxt_angle_state,
            ssm_state,
            nxt_k_prev1_state,
            nxt_k_prev2_state,
            nxt_v_prev1_state,
            nxt_v_prev2_state,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        del max_seqlen, inplace_state, kwargs

        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        angle_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )
        k_prev1_state = torch.zeros(
            (batch_size, self.nheads, self.d_state),
            device=device,
            dtype=dtype,
        )
        k_prev2_state = torch.zeros(
            (batch_size, self.nheads, self.d_state),
            device=device,
            dtype=dtype,
        )
        v_prev1_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )
        v_prev2_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (
            angle_state,
            ssm_state,
            k_prev1_state,
            k_prev2_state,
            v_prev1_state,
            v_prev2_state,
        )

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None

        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype

        if self.layer_idx not in inference_params.key_value_memory_dict:
            angle_state = torch.zeros(
                (batch_size, self.nheads, self.num_rope_angles),
                device=device,
                dtype=torch.float32,
            )
            ssm_state = torch.zeros(
                (batch_size, self.nheads, self.headdim, self.d_state),
                device=device,
                dtype=torch.float32,
            )
            k_prev1_state = torch.zeros(
                (batch_size, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )
            k_prev2_state = torch.zeros(
                (batch_size, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )
            v_prev1_state = torch.zeros(
                (batch_size, self.nheads, self.headdim),
                device=device,
                dtype=dtype,
            )
            v_prev2_state = torch.zeros(
                (batch_size, self.nheads, self.headdim),
                device=device,
                dtype=dtype,
            )

            inference_params.key_value_memory_dict[self.layer_idx] = (
                angle_state,
                ssm_state,
                k_prev1_state,
                k_prev2_state,
                v_prev1_state,
                v_prev2_state,
            )
        else:
            (
                angle_state,
                ssm_state,
                k_prev1_state,
                k_prev2_state,
                v_prev1_state,
                v_prev2_state,
            ) = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                angle_state.zero_()
                ssm_state.zero_()
                k_prev1_state.zero_()
                k_prev2_state.zero_()
                v_prev1_state.zero_()
                v_prev2_state.zero_()

        return (
            angle_state,
            ssm_state,
            k_prev1_state,
            k_prev2_state,
            v_prev1_state,
            v_prev2_state,
        )