import os
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "selective_scan_cuda" not in sys.modules:
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")

from improved_simamba_kernel import improved_simamba_siso_forward  # noqa: E402


def dtype_from_env():
    value = os.environ.get("PROFILE_DTYPE", "fp32")
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported PROFILE_DTYPE={value!r}")


def make_inputs(batch, seqlen, nheads, headdim, dtype, device):
    n_angles = headdim // 2
    value_kwargs = {"device": device, "dtype": dtype}
    coeff_kwargs = {"device": device, "dtype": torch.float32}
    return {
        "Q": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "K": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "V": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
        "ADT": -0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "DT": 0.01 + 0.2 * torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Simpson": torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Midpoint": torch.rand(batch, nheads, seqlen, **coeff_kwargs),
        "Q_bias": torch.randn(nheads, headdim, **coeff_kwargs),
        "K_bias": torch.randn(nheads, headdim, **coeff_kwargs),
        "Angles": torch.randn(batch, seqlen, nheads, n_angles, **coeff_kwargs),
        "D": torch.randn(nheads, **coeff_kwargs),
        "Z": torch.randn(batch, seqlen, nheads, headdim, **value_kwargs),
    }


def call_improved(inputs, chunk_size, num_warps, num_stages):
    return improved_simamba_siso_forward(
        inputs["Q"],
        inputs["K"],
        inputs["V"],
        inputs["ADT"],
        inputs["DT"],
        inputs["Simpson"],
        inputs["Q_bias"],
        inputs["K_bias"],
        inputs["Angles"],
        Midpoint=inputs["Midpoint"],
        D=inputs["D"],
        Z=inputs["Z"],
        chunk_size=chunk_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def main():
    batch = int(os.environ.get("PROFILE_BATCH", "2"))
    seqlen = int(os.environ.get("PROFILE_SEQLEN", "256"))
    nheads = int(os.environ.get("PROFILE_NHEADS", "32"))
    headdim = int(os.environ.get("PROFILE_HEADDIM", "64"))
    warmup_iters = int(os.environ.get("PROFILE_WARMUP", "5"))
    chunk_size = int(os.environ.get("PROFILE_CHUNK_SIZE", "64"))
    num_warps = int(os.environ.get("PROFILE_NUM_WARPS", "8"))
    num_stages = int(os.environ.get("PROFILE_NUM_STAGES", "3"))
    dtype = dtype_from_env()
    device = "cuda"

    print(
        "Allocating improved Simamba tensors: "
        f"Batch={batch}, Seq={seqlen}, Heads={nheads}, Dim={headdim}, Chunk={chunk_size}"
    )
    inputs = make_inputs(batch, seqlen, nheads, headdim, dtype, device)

    print("Running warmup...")
    for _ in range(warmup_iters):
        _ = call_improved(inputs, chunk_size, num_warps, num_stages)
    torch.cuda.synchronize()

    use_torch_profiler = os.environ.get("USE_TORCH_PROFILER", "0") == "1"
    if use_torch_profiler:
        print("Running PyTorch Profiler...")
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            _ = call_improved(inputs, chunk_size, num_warps, num_stages)
            torch.cuda.synchronize()
        return

    print("Running NVTX window for Nsight...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("Improved_Simamba_Fwd_Loop")
    _ = call_improved(inputs, chunk_size, num_warps, num_stages)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("Execution complete.")


if __name__ == "__main__":
    main()
