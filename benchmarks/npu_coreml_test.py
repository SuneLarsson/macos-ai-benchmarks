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

def create_dummy_model_vector_matrix():
    # Massive 16384 inner product (268M MACs per iteration)
    dim = 16384
    input_features = [('input', datatypes.Array(dim))]
    output_features = [('output', datatypes.Array(dim))]
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    W1 = np.random.rand(dim, dim).astype(np.float32)
    b1 = np.random.rand(dim).astype(np.float32)
    builder.add_inner_product(name='fc1', W=W1, b=b1, input_channels=dim, output_channels=dim, has_bias=True, input_name='input', output_name='output')
    
    return ct.models.MLModel(builder.spec)

def create_dummy_model_matrix_matrix():
    # Batch Multiplied Matrix (34 Billion MACs per iteration)
    # 512 batch passes simultaneously across the NPU
    batch_size = 512
    dim = 8192
    
    # Define the input feature as multi-dimensional to force the Matrix-Matrix ops.
    input_features = [('input', datatypes.Array(batch_size, dim))]
    output_features = [('output', datatypes.Array(batch_size, dim))]
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    W1 = np.random.rand(dim, dim).astype(np.float32)
    b1 = np.random.rand(dim).astype(np.float32)
    builder.add_inner_product(name='fc1', W=W1, b=b1, input_channels=dim, output_channels=dim, has_bias=True, input_name='input', output_name='output')
    
    return ct.models.MLModel(builder.spec)

def run_benchmark_variant_npu(model_factory, input_shape, variant_name, iterations=1000, seed=42, prefix="", compute_unit=ct.ComputeUnit.ALL, name="ALL", output_dir="results"):
    np.random.seed(seed)
    print(f"\n--- Starting CoreML validation benchmark: {variant_name} (ComputeUnit: {name}) ---")
    model = model_factory()
    model.save('/tmp/dummy_model.mlpackage')
    
    print(f"Loading {variant_name} model with ComputeUnit = {name}...")
    loaded_model = ct.models.MLModel('/tmp/dummy_model.mlpackage', compute_units=compute_unit)
    
    # Warmup
    np.random.seed(seed)
    dummy_input = {'input': np.random.rand(*input_shape).astype(np.float32)}
    _ = loaded_model.predict(dummy_input)
    
    print(f"Running {iterations} iterations of inference...")
    
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = f"{prefix}_" if prefix else ""
    filename = f"{output_dir}/{filename_prefix}npu_stats_{variant_name}_{name}_{time.strftime('%Y-%m-%d-%H-%M')}.json"
    
    def save_results(current_times):
        if not current_times: return
        times_arr = np.array(current_times)
        stats = {
            "benchmark": f"CoreML NPU {variant_name} ({name})",
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
            dummy_input = {'input': np.random.rand(*input_shape).astype(np.float32)}

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

    for compute_name, compute_unit in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), 
                                       ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU), 
                                       ("ALL", ct.ComputeUnit.ALL)]:
        # Vector-Matrix
        run_benchmark_variant_npu(create_dummy_model_vector_matrix, (16384,), "VecMat", iterations=args.runs, seed=args.seed, compute_unit=compute_unit, name=compute_name, output_dir=args.output_dir)
        # Matrix-Matrix
        run_benchmark_variant_npu(create_dummy_model_matrix_matrix, (512, 8192), "MatMat", iterations=args.runs, seed=args.seed, compute_unit=compute_unit, name=compute_name, output_dir=args.output_dir)

