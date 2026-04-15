#!/usr/bin/env bash
set -e

BENCH_ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --runs)
      BENCH_ARGS="$BENCH_ARGS --runs $2"
      shift 2
      ;;
    --timeout)
      BENCH_ARGS="$BENCH_ARGS --timeout $2"
      shift 2
      ;;
    --seed)
      BENCH_ARGS="$BENCH_ARGS --seed $2"
      shift 2
      ;;
    --output-dir)
      BENCH_ARGS="$BENCH_ARGS --output-dir $2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== AI Benchmark Suite Runner ==="
if [ -n "$BENCH_ARGS" ]; then
    echo "Using bench args:$BENCH_ARGS"
fi

# Get the directory of this script, which should be the NFS mount's benchmark_scripts/ dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS_NAME=$(uname -s)

if [ "$OS_NAME" = "Darwin" ]; then
    echo "--- Detected macOS Worker ---"
    
    # Step out of benchmark_scripts to create the venv directly on the NFS root
    cd ..
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing dependencies (may take a minute for coremltools)..."
    pip install numpy mlx coremltools torch transformers accelerate mlx-lm
    
    cd "$SCRIPT_DIR"
    
    echo "=================================="
    echo "--- Running CPU Baseline ---"
    python3 cpu_ml_baseline.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running GPU MLX Benchmark ---"
    python3 gpu_mlx_test.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running CoreML Hardware Validation Benchmark ---"
    python3 coreml_hardware_bench.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running Cross-Platform PyTorch Benchmark ---"
    python3 cross_platform_ai_bench.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running Cross-Platform LLM Inference Benchmark ---"
    python3 llm_inference_bench.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running NFS Data Pipeline Benchmark ---"
    python3 nfs_data_bench.py --mode stream $BENCH_ARGS
    python3 nfs_data_bench.py --mode preload $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running mlx-benchmark Suite (TristanBilot) ---"
    if [ ! -d "mlx-benchmark" ]; then
        git clone https://github.com/TristanBilot/mlx-benchmark.git
    fi
    cd mlx-benchmark
    pip install -r requirements.txt
    python3 run_benchmark.py --include_mps=True --include_mlx_gpu=True --include_mlx_cpu=False --include_cpu=False
    cd "$SCRIPT_DIR"
    
elif [ "$OS_NAME" = "Linux" ]; then
    echo "--- Detected Linux Worker ---"
    
    echo "=================================="
    echo "--- Running MLX Benchmark on Linux ---"
    python3 gpu_mlx_test.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running Cross Platform PyTorch Benchmark ---"
    python3 cross_platform_ai_bench.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running Cross-Platform LLM Inference Benchmark ---"
    python3 llm_inference_bench.py $BENCH_ARGS
    
    echo "=================================="
    echo "--- Running mlx-benchmark Suite (TristanBilot) ---"
    if [ ! -d "mlx-benchmark" ]; then
        git clone https://github.com/TristanBilot/mlx-benchmark.git
    fi
    cd mlx-benchmark
    pip install -r requirements.txt
    
    # Using mlx_gpu on linux maps to CUDA MLX
    python3 run_benchmark.py --include_mps=False --include_mlx_gpu=True --include_mlx_cpu=False --include_cuda=True --include_cpu=False
    cd "$SCRIPT_DIR"
    
else
    echo "Unknown OS: $OS_NAME"
    exit 1
fi

echo "=== AI Benchmark Suite Completed ==="
