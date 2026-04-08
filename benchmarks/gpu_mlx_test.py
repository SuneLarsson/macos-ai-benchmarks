import time
import json
import os
import numpy as np
import mlx.core as mx
import argparse
import signal
import sys

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)

def benchmark_gpu(iterations=1000, seed=42, prefix=""):
    np.random.seed(seed)
    mx.random.seed(seed)
    print("Starting MLX GPU benchmark...")
    mx.set_default_device(mx.gpu)
    print(f"MLX Default Device explicitly configured as: {mx.default_device()}")
    
    size = 4096
    print(f"Generating {size}x{size} matrices on {mx.default_device()} natively in MLX...")
    A = mx.random.uniform(shape=(size, size))
    B = mx.random.uniform(shape=(size, size))
    
    C = mx.matmul(A, B)
    mx.eval(C) # Warmup
    
    print(f"Running {iterations} iterations of matrix multiplication...")
    
    os.makedirs("results", exist_ok=True)
    filename_prefix = f"{prefix}_" if prefix else ""
    filename = f"results/{filename_prefix}gpu_stats_{int(time.time())}.json"
    
    def save_results(current_times):
        if not current_times: return
        times_arr = np.array(current_times)
        stats = {
            "benchmark": "GPU (MLX)",
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
            C = mx.matmul(A, B)
            mx.eval(C)
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
    args = parser.parse_args()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    benchmark_gpu(iterations=args.runs, seed=args.seed)
