import os
import torch

from mamba3_siso_combined import mamba3_siso_combined

def main():
    # Setup Mock Tensors
    batch, seqlen, nheads, headdim = 2, 256, 32, 64
    n_angles = headdim // 2
    device = "cuda"

    print(f"Allocating tensors: Batch={batch}, Seq={seqlen}, Heads={nheads}, Dim={headdim}")
    Q = torch.randn(batch, seqlen, nheads, headdim, device=device)
    K = torch.randn(batch, seqlen, nheads, headdim, device=device)
    V = torch.randn(batch, seqlen, nheads, headdim, device=device)
    
    ADT = torch.randn(batch, nheads, seqlen, device=device)
    DT = torch.randn(batch, nheads, seqlen, device=device)
    Simpson = torch.randn(batch, nheads, seqlen, device=device)
    
    Q_bias = torch.randn(nheads, headdim, device=device)
    K_bias = torch.randn(nheads, headdim, device=device)
    Angles = torch.randn(batch, seqlen, nheads, n_angles, device=device)

    # Warmup
    print("Running warmup...")
    for _ in range(5):
        warmup_output = mamba3_siso_combined(Q, K, V, ADT, DT, Simpson, Q_bias, K_bias, Angles)
    torch.cuda.synchronize()

    # Read the environment variable set by nsys_profiler.py
    use_torch_profiler = os.environ.get("USE_TORCH_PROFILER", "0") == "1"

    if use_torch_profiler:
        # Standalone mode: Run PyTorch Profiler
        print("Running PyTorch Profiler...")
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = mamba3_siso_combined(Q, K, V, ADT, DT, Simpson, Q_bias, K_bias, Angles)
            torch.cuda.synchronize()
    else:
        # NSYS mode: PyTorch Profiler is entirely bypassed
        print("Running NVTX window for nsys... (PyTorch Profiler DISABLED)")
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("Mamba3_Fwd_Combined_Loop")
        
        _ = mamba3_siso_combined(Q, K, V, ADT, DT, Simpson, Q_bias, K_bias, Angles)
        torch.cuda.synchronize()
        
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        
    print("Execution complete.")

if __name__ == "__main__":
    main()
