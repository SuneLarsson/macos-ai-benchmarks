import time
import json
import os
import torch
import numpy as np
import argparse
import signal
import sys

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)

def benchmark_cross_platform(iterations=1000, seed=42, prefix="", output_dir="results"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Starting Cross-Platform AI benchmark...")
    
    # Auto-detect hardware accelerator allowing identical code for Linux and Mac
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        
    print(f"Hardware explicitly acquired: {device_name}")
    
    # Mirror the AI test: large matrix multiplication
    size = 4096
    print(f"Generating {size}x{size} matrices natively on {device}... ")
    
    A = torch.rand(size, size, device=device, dtype=torch.float32)
    B = torch.rand(size, size, device=device, dtype=torch.float32)
    
    # Warmup
    C = torch.matmul(A, B)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    print(f"Running {iterations} iterations of PyTorch matrix multiplication...")
    
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = f"{prefix}_" if prefix else ""
    filename = f"{output_dir}/{filename_prefix}cross_platform_stats_{int(time.time())}.json"
    
    def save_results(current_times):
        if not current_times: return
        times_arr = np.array(current_times)
        stats = {
            "benchmark": f"PyTorch Tensor Math ({device_name})",
            "iterations_completed": len(current_times),
            "target_iterations": iterations,
            "matrix_size": size,
            "mean_s": float(np.mean(times_arr)),
            "std_s": float(np.std(times_arr)),
            "min_s": float(np.min(times_arr)),
            "max_s": float(np.max(times_arr)),
            "p10_s": float(np.percentile(times_arr, 10)),
            "p50_s": float(np.percentile(times_arr, 50)),
            "p90_s": float(np.percentile(times_arr, 90)),
            "p95_s": float(np.percentile(times_arr, 95)),
            "p99_s": float(np.percentile(times_arr, 99))
        }
        with open(filename, "w") as f:
            json.dump({"stats": stats, "raw_times_s": times_arr.tolist()}, f, indent=4)

    times = []
    try:
        for i in range(iterations):
            start = time.perf_counter()
            
            C = torch.matmul(A, B)
            
            # We must manually synchronize to measure actual compute time on asynchronous GPUs!
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
                
            times.append(time.perf_counter() - start)
            
            if (i + 1) % 50 == 0:
                save_results(times)
                print(f"Completed {i + 1}/{iterations} iterations...")
    finally:
        save_results(times)
        if times:
            print(f"Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--timeout', type=int, default=0, help='Timeout in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    benchmark_cross_platform(iterations=args.runs, seed=args.seed, output_dir=args.output_dir)
