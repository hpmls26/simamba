# Profiling

This directory contains the runnable profiling harnesses for the Simamba
project. The preferred entrypoint is the orchestration script from the
repository root:

```bash
python profiling/run_all_profiling.py --wandb
```

Use `--dry-run` to print every command without running GPU work. Use `--steps`
to run only part of the pipeline:

```bash
python profiling/run_all_profiling.py --steps kernel-nsys,kernel-correctness
python profiling/run_all_profiling.py --steps vllm,vllm-combine --wandb
```

When `--wandb` is set, all profiling runs use W&B project `profiling`.
Training scripts elsewhere in the repo still use W&B project `simamba`.

## Requirements

Install the repo with Triton support and install the profiling-only Python
packages:

```bash
python -m pip install -e '.[train,triton,causal-conv1d]' --no-build-isolation
python -m pip install vllm matplotlib wandb
```

The machine must also provide `nvidia-smi`, Nsight Systems (`nsys`), and Nsight
Compute (`ncu`). The orchestrator defaults to `/usr/local/cuda/bin/nsys` and
`/usr/local/cuda/bin/ncu`; override those with `--nsys-bin` and `--ncu-bin`.
If NCU fails with `ERR_NVGPUCTRPERM`, rerun with `--sudo-ncu` on systems where
sudo is allowed.

## Orchestration Steps

### 1. Kernel NSYS

Runs Nsight Systems once for each kernel target:

- `mamba3`: `mamba_ssm.ops.triton.mamba3.mamba3_siso_combined`
- `simamba`: `mamba_ssm.ops.triton.simamba.mamba3_siso_combined`
- `improved`: `profiling/test_kernel/improved_simamba_kernel.py`

The command writes `.nsys-rep` traces and exports report-ready CSV summaries
with `nsys stats` for CUDA kernel time, CUDA API time, NVTX ranges, and OS
runtime:

```text
profiling/results/kernel_suite/nsys/
profiling/results/kernel_suite/nsys/csv_exports/
```

### 2. Kernel NCU

Runs Nsight Compute with kernel-name filters so the report targets the useful
kernel launches instead of only the first CUDA launch. Default targets are:

- `mamba3_siso_fwd_kernel`
- `simamba_siso_fwd_kernel`
- `improved_siso_prefill_kernel`

Each target writes a `.ncu-rep` report, the direct profiler stdout CSV, an
imported raw CSV, and a text details export under:

```text
profiling/results/kernel_suite/ncu/
```

Useful overrides:

```bash
python profiling/run_all_profiling.py \
  --steps kernel-ncu \
  --ncu-seq-lens 256,1024 \
  --ncu-batch-sizes 1,2 \
  --sudo-ncu
```

### 3. Correctness

Builds a Triton-vs-PyTorch reference table for the kernel suite:

- Mamba3 forward/backward against a PyTorch Mamba3 SISO reference.
- Simamba forward/backward against the PyTorch Simamba SISO reference.
- Improved Simamba forward against the PyTorch Simamba SISO reference.

The improved kernel is a forward-only prototype, so its backward row is marked
`not_applicable`. The Mamba3 correctness check covers the core recurrence,
bias, rotary, and input-gradient path with D/Z disabled to avoid a known
small-shape D/Z backward compiler limitation.

Outputs:

```text
profiling/results/kernel_suite/kernel_correctness.csv
profiling/results/kernel_suite/kernel_correctness.md
profiling/results/kernel_suite/kernel_correctness.png
```

### 4. vLLM Model Sweep

Runs repeated vLLM measurements for:

- Mamba2: `soumil1/mamba2-10m-slimpajama-500m`
- Improved Simamba: `outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export`

The sweep logs raw samples and summary statistics for TTFT, TPOT, end-to-end
tok/s, decode-loop tok/s, prefill-probe latency, requests/s, GPU memory peak
and delta, model-load memory delta, prefix caching on/off, and 512-token
repeated-prefix prompts.

The sweep exits non-zero if any model/load/generation row fails, so report runs
do not silently upload partial `load_failed` data. Use `--allow-failures` only
for exploratory debugging.

Outputs:

```text
profiling/results/vllm_mamba2_summary.csv
profiling/results/vllm_mamba2_raw.csv
profiling/results/vllm_mamba2.png
profiling/results/vllm_improved_simamba_summary.csv
profiling/results/vllm_improved_simamba_raw.csv
profiling/results/vllm_improved_simamba.png
```

The improved Simamba run uses the vLLM fork at `/tmp/hpmls26_vllm` by default
and passes `--simamba-native-backend triton`. Override with:

```bash
python profiling/run_all_profiling.py \
  --steps vllm \
  --improved-model /path/to/vllm_export \
  --simamba-vllm-fork /path/to/vllm
```

### 5. Combined vLLM Plot and Bundle

`vllm-combine` merges the Mamba2 and improved Simamba CSVs and creates a single
comparison plot:

```text
profiling/results/vllm_mamba2_vs_improved_summary.csv
profiling/results/vllm_mamba2_vs_improved_raw.csv
profiling/results/vllm_mamba2_vs_improved.png
```

When `--wandb` is set, the final `bundle` step uploads the kernel traces,
CSV exports, correctness tables/plots, vLLM raw data, and matplotlib plots as a
single W&B artifact.

## Common Commands

Full local run without W&B:

```bash
python profiling/run_all_profiling.py
```

Full W&B run with NCU through sudo:

```bash
python profiling/run_all_profiling.py --wandb --sudo-ncu
```

Fast command inspection:

```bash
python profiling/run_all_profiling.py --dry-run --wandb
```
