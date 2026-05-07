import os


import time
import argparse
import wandb
from vllm import LLM, SamplingParams


def mamba2_hf_overrides(config):
    """Fill vLLM's generic architecture fields for HF Mamba-family configs."""
    if getattr(config, "model_type", None) in {"mamba2", "simamba"}:
        num_heads = getattr(config, "num_heads", None)
        if num_heads is None and getattr(config, "model_type", None) == "simamba":
            ssm_cfg = getattr(config, "ssm_cfg", {}) or {}
            headdim = ssm_cfg.get("headdim")
            hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
            if headdim and hidden_size:
                num_heads = hidden_size // headdim
        if num_heads is not None:
            if not hasattr(config, "num_attention_heads"):
                config.num_attention_heads = num_heads
            if not hasattr(config, "num_key_value_heads"):
                config.num_key_value_heads = num_heads
        if not hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = 1024
    return config



def main(args):
    # 1. Setup Profiler Directory
    # vLLM looks for this environment variable to enable PyTorch profiling
    trace_dir = f"./vllm_traces_{int(time.time())}"
    os.makedirs(trace_dir, exist_ok=True)

    # 2. Initialize Weights & Biases
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "profiling"),
        job_type="profile",
        name=args.wandb_name,
        config={
            "model": args.model,
            "max_tokens": args.max_tokens,
            "tensor_parallel_size": args.tensor_parallel,
            
        }
    )

    print(f"Loading model: {args.model}...")
    
    # 3. Initialize vLLM Engine
    # trust_remote_code=True is often required for state-space models like Mamba
    llm = LLM(
        model=args.model,
        tokenizer="EleutherAI/gpt-neox-20b",
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=False,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.25,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        hf_overrides=mamba2_hf_overrides,
        profiler_config={
            "profiler": "torch", 
            "torch_profiler_dir": trace_dir
        }
    )

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy decoding for consistent profiling
        max_tokens=args.max_tokens
    )

    prompts = [
        "The future of high-performance machine learning is",
        "Explain the architecture of a state space model:",
    ]

    # 4. Warmup Step
    # Always run a warmup to compile CUDA graphs and allocate KV cache
    # Otherwise, your profile trace will just be initialization overhead.
    print("Running warmup...")
    llm.generate(prompts, sampling_params, use_tqdm=False)

    # 5. The Actual Profiling Step
    print("Starting profiler...")
    llm.start_profile()
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.perf_counter()
    
    llm.stop_profile()
    print("Profiling complete.")

    # 6. Calculate & Log High-Level Metrics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    elapsed_time = end_time - start_time
    tokens_per_sec = total_tokens / elapsed_time

    wandb.log({
        "generation_time_sec": elapsed_time,
        "total_generated_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec
    })

    # 7. Upload the Trace to W&B Artifacts
    # vLLM drops a .json trace file in the VLLM_TORCH_PROFILER_DIR
    print("Uploading trace to W&B...")
    artifact = wandb.Artifact(
        name=f"vllm_trace_{run.id}", 
        type="profiling_trace",
        description=f"PyTorch profile trace for {args.model}"
    )
    artifact.add_dir(trace_dir)
    run.log_artifact(artifact)

    wandb.finish()
    print("Done! You can view your trace in the W&B dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile vLLM and log to W&B")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b-hf", help="HuggingFace model ID")
    parser.add_argument("--max_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--wandb_name", type=str, default=None, help="Optional W&B run name")
    
    args = parser.parse_args()
    main(args)
