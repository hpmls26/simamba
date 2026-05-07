# Profiling

This directory keeps only runnable profiling harnesses and the local kernel
sources they import.

## Final Report Pipeline

Run the full final-report profiling reproduction from the repository root:

```bash
python profiling/run_all_profiling.py --wandb
```

This runs the report NSYS traces, vLLM repeated measurements, vLLM combination
plot, Triton-vs-PyTorch correctness table, and final W&B artifact bundle. Use
`--dry-run` to inspect the underlying commands before launching long GPU jobs.
Use `--steps nsys,vllm,combine,correctness` to run a subset, or
`--steps kernel-sweep,plots` for the optional sweep helpers.

The script defaults to W&B project `profiling` when `--wandb` is set. Training
jobs elsewhere in the repository still use project `simamba`.

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

The scripts default W&B logging to project `profiling`.

## Report Data Sweeps

Kernel latency, memory, optional backward timing, and optional correctness:

```bash
python profiling/kernel_sweep.py \
  --seq-lens 128,256,512,1024 \
  --batch-sizes 1,2,4 \
  --iters 20 \
  --compare \
  --wandb \
  --out results/kernel_sweep.csv
```

vLLM throughput and latency sweeps:

```bash
python profiling/vllm_sweep.py \
  --prompt-words 32,128,512 \
  --batch-sizes 1,2,4 \
  --max-tokens 32,128 \
  --warmup 1 \
  --repeats 5 \
  --wandb \
  --out results/vllm_sweep.csv
```

The sweep writes a summary CSV, a raw-samples CSV, and a matplotlib summary
plot. GPU memory is sampled from global device usage via `nvidia-smi`, so it
captures vLLM worker-process memory instead of parent-process PyTorch
allocations. Summary rows include mean, median, p95, std, min, and max for
TTFT, TPOT, total tok/s, explicit decode-loop timing, prefill-probe timing,
and GPU memory.

Create report plots after the sweeps:

```bash
python profiling/plot_profile_results.py \
  --kernel-csv results/kernel_sweep.csv \
  --vllm-csv results/vllm_sweep.csv \
  --out-dir results/plots \
  --wandb
```

The plotting script writes:

- `kernel_latency.png`
- `kernel_memory.png`
- `vllm_throughput.png`
- `vllm_latency.png`

## Current Report Artifacts

- vLLM Mamba2/Simamba TTFT, TPOT, tok/s, 512-token repeated-prefix rows, decode-loop timing, prefill-probe timing, and GPU-memory summaries: `results/vllm_repeated_combined_summary.csv`
- Raw repeated vLLM samples: `results/vllm_repeated_combined_raw.csv`
- vLLM matplotlib comparison plot: `results/vllm_repeated_combined_summary.png`
- Triton-vs-PyTorch forward/backward correctness: `results/kernel_correctness_reference.csv` and `results/kernel_correctness_reference.md`

## W&B Full Rerun

All report scripts support `--wandb`, but the preferred full rerun entrypoint
is the orchestrator:

```bash
python profiling/run_all_profiling.py --wandb
```

The W&B artifact names are `nsys_kernel_reports`, `vllm_sweep_results`,
`vllm_combined_repeated_profile`, `kernel_correctness_reference`, and
`profiling_full_run_bundle`.
