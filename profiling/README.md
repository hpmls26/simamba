# Profiling

This directory keeps only runnable profiling harnesses and the local kernel
sources they import.

## Kernel NSYS Profiles

Run from each kernel directory so local imports resolve correctly:

```bash
cd profiling/simamba
python nsys_profiler.py
```

```bash
cd profiling/mamba3
python nsys_profiler.py
```

`simamba/` profiles the Simamba SISO combined kernel. `mamba3/` profiles the
Euler/original discretization SISO combined kernel.

## vLLM Profiles

Run from `profiling/`:

```bash
python vllm_profiler.py --model state-spaces/mamba-130m-hf --wandb_name vllm_mamba_130m_hf
python vllm_profiler.py --model benchang1110/mamba2-130m-hf --wandb_name vllm_mamba2_130m_hf
python vllm_profiler.py --model soumil1/mamba2-10m-slimpajama-500m --wandb_name vllm_soumil1_mamba2_10m_slimpajama_500m
```

The scripts default W&B logging to project `ssb2234-columbia`.
