#!/usr/bin/env bash
set -e
BENCH_ARGS=""
OUTPUT_DIR="results"
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
      OUTPUT_DIR="$2"
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

if [[ "$OUTPUT_DIR" != /* ]]; then
    ABS_OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
else
    ABS_OUTPUT_DIR="$OUTPUT_DIR"
fi
mkdir -p "$ABS_OUTPUT_DIR"

# Override HuggingFace cache to always map to the local execution environment's incredibly fast /tmp storage!
# We intentionally avoid $SCRIPT_DIR (Ceph/NFS) because network-attached storage is notoriously slow for 
# streaming huge multi-gigabyte model weights directly into unified memory, and NFS POSIX filelocks
# frequently deadlock macOS clients independently.
export HF_HOME="/tmp/ai_bench_hf_cache"
mkdir -p "$HF_HOME"

OS_NAME=$(uname -s)

if [ "$OS_NAME" = "Darwin" ]; then
    echo "--- Detected macOS Worker ---"
    
    # Create the venv in a unique /private/tmp directory to avoid NFS locking and concurrent run collisions
    VENV_DIR=$(mktemp -d -t ai_bench_venv)
    echo "Setting up Python virtual environment in $VENV_DIR..."
    trap 'echo "Cleaning up $VENV_DIR..."; rm -rf "$VENV_DIR"' EXIT
    
    if ! python3 -m venv "$VENV_DIR"; then
        echo "venv creation failed. Attempting with --without-pip..."
        python3 -m venv --without-pip "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    echo "Installing dependencies (may take a minute for coremltools)..."
    pip install numpy mlx coremltools torch torchvision transformers accelerate mlx-lm "urllib3<2"
    
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
    cd "$VENV_DIR"
    if [ ! -d "mlx-benchmark" ]; then
        git clone https://github.com/TristanBilot/mlx-benchmark.git
    fi
    cd mlx-benchmark
    pip install -r requirements.txt torchvision
    
    cd mlx_benchmark
    python3 run_benchmark.py --include_mps=True --include_mlx_gpu=True --include_mlx_cpu=False --include_cpu=False | tee "$ABS_OUTPUT_DIR/mlx_benchmark_suite_mac.txt"
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
    # Clone to a safe localized tmp directory to avoid bare-metal Ceph concurrent locking natively
    WORK_DIR=$(mktemp -d)
    cd "$WORK_DIR"
    
    if [ ! -d "mlx-benchmark" ]; then
        git clone https://github.com/TristanBilot/mlx-benchmark.git
    fi
    cd mlx-benchmark
    pip install -r requirements.txt torchvision
    
    cd mlx_benchmark
    # Using mlx_gpu on linux maps to CUDA MLX
    python3 run_benchmark.py --include_mps=False --include_mlx_gpu=True --include_mlx_cpu=False --include_cuda=True --include_cpu=False | tee "$ABS_OUTPUT_DIR/mlx_benchmark_suite_linux.txt"
    cd "$SCRIPT_DIR"
    
else
    echo "Unknown OS: $OS_NAME"
    exit 1
fi

echo "=== AI Benchmark Suite Completed ==="
