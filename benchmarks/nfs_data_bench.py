import argparse
import time
import os
import shutil
import numpy as np
import mlx.core as mx
import json
import signal
import sys

DATASET_SIZE_MB = 1000  # 1GB dummy dataset

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)
DATASET_FILE = "dummy_nfs_dataset.npy"
LOCAL_TMP_DIR = "/private/tmp/mlx_nfs_test_data"

def generate_dataset(filepath, seed=42):
    np.random.seed(seed)
    print(f"Generating {DATASET_SIZE_MB}MB dummy dataset at {filepath}...")
    # 1 float32 = 4 bytes. 1GB = 250 million floats
    num_floats = (DATASET_SIZE_MB * 1024 * 1024) // 4
    data = np.random.rand(num_floats).astype(np.float32)
    np.save(filepath, data)
    print("Dataset generated successfully.")

def simulate_training(filepath, iterations=50, seed=42):
    mx.random.seed(seed)
    print(f"Loading data into RAM from {filepath}...")
    start_load = time.perf_counter()
    data = np.load(filepath)
    load_time = time.perf_counter() - start_load
    print(f"Loaded {len(data)*4/1024/1024:.2f} MB in {load_time:.2f}s")
    
    # Simulate batch training (transferring to MLX device and doing math)
    mx_data = mx.array(data)
    batch_size = 1024 * 1024 * 2 # 8MB batches
    num_batches = len(data) // batch_size
    
    print(f"Simulating {iterations} epochs computing {num_batches} batches...")
    
    weights = mx.random.uniform(shape=(batch_size,))
    
    start_train = time.perf_counter()
    for __ in range(iterations):
        for i in range(num_batches):
            batch = mx_data[i*batch_size : (i+1)*batch_size]
            # Mock dense computation to prevent graph from optimizing out
            loss = mx.sum(batch * weights)
            mx.eval(loss)
            
    train_time = time.perf_counter() - start_train
    print(f"Mock training loops completed in {train_time:.2f}s")
    
    return load_time, train_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['stream', 'preload'], required=True)
    parser.add_argument('--runs', type=int, default=50, help='Number of iterations (epochs)')
    parser.add_argument('--timeout', type=int, default=0, help='Timeout in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)
    
    if not os.path.exists(DATASET_FILE):
        generate_dataset(DATASET_FILE, seed=args.seed)
        
    stats = {
        "mode": args.mode,
        "dataset_size_mb": DATASET_SIZE_MB,
        "epochs": args.runs
    }
    
    if args.mode == 'preload':
        print(f"=== NATIVE PRELOAD MODE (NFS -> SSD -> RAM) ===")
        os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
        local_filepath = os.path.join(LOCAL_TMP_DIR, DATASET_FILE)
        
        print(f"Copying {DATASET_FILE} (NFS) to {local_filepath} (Local SSD)...")
        start_copy = time.perf_counter()
        shutil.copy2(DATASET_FILE, local_filepath)
        copy_time = time.perf_counter() - start_copy
        print(f"Copy completed in {copy_time:.2f}s ({(DATASET_SIZE_MB/copy_time):.2f} MB/s)")
        
        stats['nfs_to_ssd_copy_time_s'] = copy_time
        load_time, train_time = simulate_training(local_filepath, iterations=args.runs, seed=args.seed)
        stats['read_to_ram_time_s'] = load_time
        stats['compute_training_time_s'] = train_time
        stats['total_pipeline_time_s'] = copy_time + load_time + train_time
        
        # Cleanup local
        os.remove(local_filepath)
        
    elif args.mode == 'stream':
        print(f"=== NETWORK STREAM MODE (NFS -> RAM) ===")
        stats['nfs_to_ssd_copy_time_s'] = 0.0
        
        load_time, train_time = simulate_training(DATASET_FILE, iterations=args.runs, seed=args.seed)
        stats['read_to_ram_time_s'] = load_time
        stats['compute_training_time_s'] = train_time
        stats['total_pipeline_time_s'] = load_time + train_time

    print("--- Benchmark Results ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/nfs_data_bench_{args.mode}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
