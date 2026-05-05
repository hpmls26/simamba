import wandb
import subprocess
import os
import sys

def main():
    # Force W&B to completely disable its background hardware polling
    os.environ["WANDB_DISABLE_SYSTEM_METRICS"] = "true"
    
    # 1. Initialize W&B
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ssb2234-columbia"),
        job_type="kernel_profiling",
        name="mamba_combined"
    )
    
    nsys_output = f"nsys_trace_{run.id}"
    
    # 2. Construct the nsys command
    cmd = [
        "/usr/local/cuda/bin/nsys", "profile",
        "--trace=cuda,nvtx,osrt", 
        "--capture-range=cudaProfilerApi", 
        f"--output={nsys_output}",
        "--force-overwrite=true",
        sys.executable, "kernel_profiler.py"
    ]
    
    # Force disable PyTorch's internal profiler
    env = os.environ.copy()
    env["DISABLE_KINETO"] = "1"
    env["USE_TORCH_PROFILER"] = "0"
    
    print("Launching NSYS profiling...")
    
    # Run the subprocess and catch the exit code
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Notice] nsys command exited with code {e.returncode}.")
        print("Checking if the trace file was successfully generated anyway...")
    
    # 3. Check for the file and upload
    trace_file = f"{nsys_output}.nsys-rep"
    
    if os.path.exists(trace_file):
        print(f"\n✅ Success! Found {trace_file}. Uploading to W&B...")
        nsys_artifact = wandb.Artifact(
            name=f"nsys_bwd_trace_{run.id}",
            type="nsys_trace",
            description="Low-level NSYS timeline for Simamba backward step"
        )
        nsys_artifact.add_file(trace_file)
        run.log_artifact(nsys_artifact)
    else:
        print(f"\n❌ Error: Profiling failed and {trace_file} was not generated.")
    
    wandb.finish()
    print("\nScript complete. If successful, download the .nsys-rep file from W&B.")

if __name__ == "__main__":
    main()
