#!/usr/bin/env bash
set -e

# ==============================================================================
# Linux Dedicated AI Benchmark Runner
# ==============================================================================

BENCH_ARGS=""
OUTPUT_DIR="results_linux"

USE_NATIVE=false

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
    --native)
      USE_NATIVE=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== Linux LLM Benchmark Suite ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$OUTPUT_DIR" != /* ]]; then
    ABS_OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
else
    ABS_OUTPUT_DIR="$OUTPUT_DIR"
fi
mkdir -p "$ABS_OUTPUT_DIR"

# Ensure we use fast tmp cache for HuggingFace if available
export HF_HOME="/tmp/ai_bench_hf_cache"
mkdir -p "$HF_HOME"

# Bypass getpass.getuser() crashes on Kubernetes when running as an anonymous UID
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"
export TRITON_CACHE_DIR="/tmp/triton_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Torchvision comes pre-installed broken on some PyTorch containers and crashes Transformers.
# Since we only do text LLM inference, we can safely rip it out to unblock execution!
pip uninstall -y torchvision > /dev/null 2>&1

echo "=================================="
if [ "$USE_NATIVE" = true ]; then
    echo "--- Running Dedicated Linux LLM Benchmark (NATIVE UNQUANTIZED) ---"
    python3 linux_inference_native.py $BENCH_ARGS
else
    echo "--- Running Dedicated Linux LLM Benchmark (4-BIT) ---"
    python3 linux_inference.py $BENCH_ARGS
fi

echo "=== Linux Benchmark Suite Completed ==="
