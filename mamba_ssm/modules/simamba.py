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
    simamba_trapezoid_siso_combined,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (
    mamba3_siso_combined as simamba_triton_siso_combined,
)
from mamba_ssm.ops.triton.simamba.mamba3_siso_step import mamba3_siso_step as simamba_triton_siso_step


SIMAMBA_BACKEND_REFERENCE = "reference"
SIMAMBA_BACKEND_TRITON = "triton"
SIMAMBA_BACKEND_IMPROVED = "improved"
SIMAMBA_SUPPORTED_BACKENDS = (
    SIMAMBA_BACKEND_REFERENCE,
    SIMAMBA_BACKEND_TRITON,
    SIMAMBA_BACKEND_IMPROVED,
)
SIMAMBA_DISCRETIZATION_SIMPSON = "simpson"
SIMAMBA_DISCRETIZATION_TRAPEZOID = "trapezoid"
SIMAMBA_SUPPORTED_DISCRETIZATIONS = (
    SIMAMBA_DISCRETIZATION_SIMPSON,
    SIMAMBA_DISCRETIZATION_TRAPEZOID,
)


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
        dt_limit=(0.001, 0.1),
        A_floor=1e-4,
        A_max=16.0,
        d_conv=0,
        conv_bias=True,
        conv_init=None,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
        recompute_chunk_size=None,
        use_midpoint_control=False,
        control_logit_offset=0.0,
        midpoint_logit_offset=0.0,
        simpson_correction_scale=1.0,
        discretization=SIMAMBA_DISCRETIZATION_SIMPSON,
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
        self.recompute_chunk_size = recompute_chunk_size if recompute_chunk_size is not None else chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.A_max = float(A_max)
        self.d_conv = int(d_conv)
        if self.d_conv < 0:
            raise ValueError(f"d_conv must be non-negative, got {d_conv!r}.")
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = False
        self.mimo_rank = 1
        self.use_midpoint_control = use_midpoint_control
        self.control_logit_offset = float(control_logit_offset)
        self.midpoint_logit_offset = float(midpoint_logit_offset)
        simpson_correction_scale = float(simpson_correction_scale)
        if not math.isfinite(simpson_correction_scale):
            raise ValueError(f"simpson_correction_scale must be finite, got {simpson_correction_scale!r}.")
        if simpson_correction_scale < 0.0 or simpson_correction_scale > 1.0:
            raise ValueError(
                f"simpson_correction_scale must be in [0, 1], got {simpson_correction_scale!r}."
            )
        self.register_buffer(
            "simpson_correction_scale",
            torch.tensor(simpson_correction_scale, dtype=torch.float32, device=device),
            persistent=False,
        )
        if discretization not in SIMAMBA_SUPPORTED_DISCRETIZATIONS:
            raise ValueError(
                f"Unsupported discretization={discretization!r}. "
                f"Expected one of {SIMAMBA_SUPPORTED_DISCRETIZATIONS}."
            )
        if discretization == SIMAMBA_DISCRETIZATION_TRAPEZOID and self.use_midpoint_control:
            raise ValueError("Trapezoid baseline does not use midpoint control.")
        self.discretization = discretization
        if len(dt_limit) != 2:
            raise ValueError(f"dt_limit must be a (min, max) pair, got {dt_limit!r}.")
        dt_limit_min, dt_limit_max = float(dt_limit[0]), float(dt_limit[1])
        if dt_limit_min < 0.0 or dt_limit_max < dt_limit_min:
            raise ValueError(
                f"Invalid dt_limit={dt_limit!r}; expected 0 <= min <= max."
            )
        self.dt_limit = (dt_limit_min, dt_limit_max)
        if self.A_max <= 0.0:
            raise ValueError(f"A_max must be positive, got {A_max!r}.")
        if math.isfinite(self.A_max) and self.A_max < self.A_floor:
            raise ValueError(
                f"A_max={self.A_max} must be >= A_floor={self.A_floor}."
            )
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

        # Order: [z, x, B, C, dd_dt, dd_A, coefficient, (optional midpoint), angle]
        num_head_controls = 4 if self.use_midpoint_control else 3
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + num_head_controls * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        if self.d_conv > 0:
            conv_dim = self.d_inner + 2 * self.d_state * self.num_bc_heads
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=self.d_conv,
                groups=conv_dim,
                padding=self.d_conv - 1,
                **factory_kwargs,
            )
            if conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -conv_init, conv_init)
            self.act = nn.SiLU()
        else:
            self.conv1d = None
            self.act = nn.Identity()

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

    def _bounded_dynamics(self, dd_A, dd_dt):
        _A = -F.softplus(dd_A.to(torch.float32))
        if math.isfinite(self.A_max):
            _A = torch.clamp(_A, min=-self.A_max, max=-self.A_floor)
        else:
            _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        if self.dt_limit != (0.0, float("inf")):
            DT = DT.clamp(min=self.dt_limit[0], max=self.dt_limit[1])
        return _A, DT

    @torch.no_grad()
    def set_simpson_correction_scale(self, scale: float) -> None:
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError(f"Simpson correction scale must be finite, got {scale!r}.")
        self.simpson_correction_scale.fill_(max(0.0, min(1.0, scale)))

    def get_simpson_correction_scale(self) -> float:
        return float(self.simpson_correction_scale.detach().cpu().item())

    def _scale_simpson_correction(self, simpson: torch.Tensor) -> torch.Tensor:
        if self.discretization != SIMAMBA_DISCRETIZATION_SIMPSON:
            return simpson
        scale = self.simpson_correction_scale.to(device=simpson.device, dtype=torch.float32)
        return simpson * scale

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        del seq_idx
        if cu_seqlens is not None:
            raise NotImplementedError("Simamba does not support cu_seqlens / variable-length mode.")
        if u.dim() != 3:
            raise ValueError(f"Expected input shape (batch, seqlen, d_model), got {u.shape}.")
        if inference_params is not None and self.d_conv > 0:
            raise NotImplementedError("Incremental decoding is not implemented for Simamba local convolution.")

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
                if self.discretization == SIMAMBA_DISCRETIZATION_TRAPEZOID:
                    raise NotImplementedError("Incremental decoding is not implemented for the trapezoid baseline.")
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

        if self.conv1d is not None:
            xBC = torch.cat([x, B, C], dim=-1)
            xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
            if self.d_conv > 1:
                xBC = xBC[:, :-(self.d_conv - 1)]
            xBC = self.act(xBC)
            x, B, C = torch.split(
                xBC,
                [
                    self.d_inner,
                    self.d_state * self.num_bc_heads,
                    self.d_state * self.num_bc_heads,
                ],
                dim=-1,
            )

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        simpson = rearrange(torch.sigmoid(simpson + self.control_logit_offset), "b l h -> b h l")
        simpson = self._scale_simpson_correction(simpson)
        midpoint = (
            rearrange(torch.sigmoid(midpoint + self.midpoint_logit_offset), "b l h -> b h l")
            if midpoint is not None
            else None
        )

        _A, DT = self._bounded_dynamics(dd_A, dd_dt)
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

        if self.discretization == SIMAMBA_DISCRETIZATION_TRAPEZOID:
            if self.simamba_backend != SIMAMBA_BACKEND_REFERENCE:
                raise NotImplementedError(
                    "The matched trapezoid Simamba baseline currently uses the reference backend. "
                    "Set --simamba-backend reference for --simamba-discretization trapezoid."
                )
            y = simamba_trapezoid_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=simpson,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                Input_States=input_states,
                return_final_states=input_states is not None,
                cu_seqlens=cu_seqlens,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k1, last_k2, last_v1, last_v2 = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_prev1_state.copy_(last_k1)
                k_prev2_state.copy_(last_k2)
                v_prev1_state.copy_(last_v1)
                v_prev2_state.copy_(last_v2)
        elif self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
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
            y_tuple = simamba_triton_siso_combined(
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
                recompute_chunk_size=self.recompute_chunk_size,
                Initial_States=input_states,
                return_final_states=input_states is not None,
                cu_seqlens=cu_seqlens,
                use_improved_kernel=self.simamba_backend == SIMAMBA_BACKEND_IMPROVED,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k1, last_k2, last_v1, last_v2 = y_tuple
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_prev1_state.copy_(last_k1)
                k_prev2_state.copy_(last_k2)
                v_prev1_state.copy_(last_v1)
                v_prev2_state.copy_(last_v2)
            else:
                y = y_tuple

        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)

        out = self.out_proj(y.to(x.dtype))
        return out

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, simpson_proj, angle_proj, midpoint_proj=None):
        _A, DT = self._bounded_dynamics(A_proj, dd_dt)
        simpson = torch.sigmoid(simpson_proj + self.control_logit_offset)
        simpson = self._scale_simpson_correction(simpson)
        midpoint = torch.sigmoid(midpoint_proj + self.midpoint_logit_offset) if midpoint_proj is not None else None

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

        if self.d_conv > 0:
            raise NotImplementedError("Incremental decoding is not implemented for Simamba local convolution.")

        if self.num_bc_heads != 1:
            raise NotImplementedError(
                "Simamba incremental decode currently supports ngroups=1 only. "
                f"Received ngroups={self.num_bc_heads}."
            )

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
                Output_States=(
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

        if nxt_angle_state is not angle_state:
            angle_state.copy_(nxt_angle_state)
        if nxt_ssm_state is not ssm_state:
            ssm_state.copy_(nxt_ssm_state)
        if nxt_k_prev1_state is not k_prev1_state:
            k_prev1_state.copy_(nxt_k_prev1_state)
        if nxt_k_prev2_state is not k_prev2_state:
            k_prev2_state.copy_(nxt_k_prev2_state)
        if nxt_v_prev1_state is not v_prev1_state:
            v_prev1_state.copy_(nxt_v_prev1_state)
        if nxt_v_prev2_state is not v_prev2_state:
            v_prev2_state.copy_(nxt_v_prev2_state)

        return (
            out,
            nxt_angle_state,
            nxt_ssm_state,
            nxt_k_prev1_state,
            nxt_k_prev2_state,
            nxt_v_prev1_state,
            nxt_v_prev2_state,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        del max_seqlen, inplace_state, kwargs

        if self.d_conv > 0:
            raise NotImplementedError("Inference cache is not implemented for Simamba local convolution.")

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
