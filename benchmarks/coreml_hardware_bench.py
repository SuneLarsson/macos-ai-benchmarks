import time
import os
import json
import numpy as np
import coremltools as ct
import argparse
import signal
import sys
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)

def create_dummy_model():
    # Create a simple neural network: an MLP with one hidden layer
    # Size increased to 2048 to ensure ANE invocation (ANE ignores very tiny models due to overhead)
    input_features = [('input', datatypes.Array(2048))]
    output_features = [('output', datatypes.Array(2048))]
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    W1 = np.random.rand(2048, 2048).astype(np.float32)
    b1 = np.random.rand(2048).astype(np.float32)
    builder.add_inner_product(name='fc1', W=W1, b=b1, input_channels=2048, output_channels=2048, has_bias=True, input_name='input', output_name='output')
    
    mlmodel = ct.models.MLModel(builder.spec)
    return mlmodel

def benchmark_coreml(iterations=1000, seed=42, prefix="", compute_unit=ct.ComputeUnit.ALL, name="ALL", output_dir="results"):
    np.random.seed(seed)
    print(f"\n--- Starting CoreML validation benchmark (ComputeUnit: {name}) ---")
    model = create_dummy_model()
    model.save('/tmp/dummy_model.mlpackage')
    
    print(f"Loading model with ComputeUnit = {name}...")
    loaded_model = ct.models.MLModel('/tmp/dummy_model.mlpackage', compute_units=compute_unit)
    # Warmup
    np.random.seed(seed)
    dummy_input = {'input': np.random.rand(2048).astype(np.float32)}
    _ = loaded_model.predict(dummy_input)
    
    print(f"Running {iterations} iterations...")
    
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = f"{prefix}_" if prefix else ""
    filename = f"{output_dir}/{filename_prefix}coreml_stats_{name}_{int(time.time())}.json"
    
    def save_results(current_times):
        if not current_times: return
        times_arr = np.array(current_times)
        stats = {
            "benchmark": f"CoreML ({name})",
            "iterations_completed": len(current_times),
            "target_iterations": iterations,
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
            current_seed = seed + i
            np.random.seed(current_seed)
            dummy_input = {'input': np.random.rand(2048).astype(np.float32)}

            start = time.perf_counter()
            _ = loaded_model.predict(dummy_input)
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

    # Run explicitly on CPU to establish a CoreML baseline
    benchmark_coreml(iterations=args.runs, seed=args.seed, compute_unit=ct.ComputeUnit.CPU_ONLY, name="CPU_ONLY", output_dir=args.output_dir)
    # Run with CPU AND GPU to track GPU performance impact within CoreML
    benchmark_coreml(iterations=args.runs, seed=args.seed, compute_unit=ct.ComputeUnit.CPU_AND_GPU, name="CPU_AND_GPU", output_dir=args.output_dir)
    # Run with ALL (Which forces ANE if available)
    benchmark_coreml(iterations=args.runs, seed=args.seed, compute_unit=ct.ComputeUnit.ALL, name="ALL", output_dir=args.output_dir)
