import os
import torch
from mamba3_siso_bwd import compute_dcoeffs

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
    grad_out = torch.randn(batch, seqlen, nheads, headdim, device=device)

    # Warmup
    print("Running warmup...")
    for _ in range(5):
        _ = compute_dcoeffs(Q, K, V, ADT, DT, Simpson, Q_bias, K_bias, Angles, grad_out)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("Simamba_Bwd_Step_Loop")
    
    _ = compute_dcoeffs(Q, K, V, ADT, DT, Simpson, Q_bias, K_bias, Angles, grad_out)
    torch.cuda.synchronize()
    
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
        
    print("Execution complete.")

if __name__ == "__main__":
    main()