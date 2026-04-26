__version__ = "2.3.1"

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "Mamba3",
    "Simamba",
    "MambaLMHeadModel",
]


def __getattr__(name):
    if name in {"selective_scan_fn", "mamba_inner_fn"}:
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

        return {
            "selective_scan_fn": selective_scan_fn,
            "mamba_inner_fn": mamba_inner_fn,
        }[name]
    if name == "Mamba":
        from mamba_ssm.modules.mamba_simple import Mamba

        return Mamba
    if name == "Mamba2":
        from mamba_ssm.modules.mamba2 import Mamba2

        return Mamba2
    if name == "Mamba3":
        from mamba_ssm.modules.mamba3 import Mamba3

        return Mamba3
    if name == "Simamba":
        from mamba_ssm.modules.simamba import Simamba

        return Simamba
    if name == "MambaLMHeadModel":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
