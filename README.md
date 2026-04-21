# macos-exec-kubelet AI Benchmarking Suite

This directory contains an AI/ML-focused benchmarking suite designed to expose the exact performance and hardware acceleration differences between native macOS execution (`macos-exec-kubelet`) and a virtualized macOS environment (`macOS-vz-kubelet`).

## What the Suite Tests

1. **Apple Neural Engine (NPU) validation:** Uses `coremltools` to target the ANE. On `macos-exec-kubelet`, this runs with blistering speed. On `macOS-vz-kubelet`, the NPU is unavailable, triggering an automatic fallback to the CPU/GPU resulting in measurable latency drops.
2. **GPU Performance:** Uses Apple's `mlx` framework to measure GPU compute capabilities for AI workloads like LLMs.
3. **CPU Baseline:** Uses `numpy` to measure raw floating-point calculations restricted entirely to the CPU cores.
4. **All-Systems-Go Stress Test:** Uses multiprocessing to run all three simultaneously, proving thermal and scheduling capabilities.
5. **Storage Overhead (Model/Dataset Loading):** Uses `fio` to simulate the network latency and throughput of reading large model weights and datasets over NFS.

## Prerequisites

- **Python Environment:** Python 3.10+ is required to ensure compatibility with `coremltools` and `mlx`.
- **Python Packages:** To run or test the scripts locally, you need the following dependencies:
  ```bash
  pip install numpy mlx coremltools torch transformers accelerate mlx-lm
  ```
- **Python Scripts:** The Python files located in the `benchmarks/` folder must be uploaded to your Ceph/NFS persistent volume so that the jobs can access them without building container images.
- **fio:** To run the macOS Storage Benchmark, you must install `fio` directly on the macOS runner:
  ```bash
  brew install fio
  ```

## Running the Benchmarks

### Running Locally (Without Kubernetes)

While the expected usage is via the orchestrated Kubernetes Jobs, you can run the suite directly on your hardware to validate the Python scripts and environment.

**On macOS (Apple Silicon):**
```bash
cd benchmarks/
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install the required packages
pip install numpy mlx coremltools torch transformers accelerate mlx-lm
# Execute the unified test suite
bash run_ai_suite.sh

# Alternatively, run only specific benchmarks:
# bash run_ai_suite.sh --test "coreml,llm"
```

**On Linux (NVIDIA CUDA):**
We recommend using the provided `Dockerfile.linux` to establish the correct PyTorch and CUDA layer.
```bash
cd benchmarks/
# The base image already has PyTorch, but we need MLX and LLM libraries
pip install mlx transformers accelerate mlx-lm
# Execute the suite (it will detect Linux and pivot to CUDA testing)
bash run_ai_suite.sh

# Alternatively, run only specific benchmarks:
# bash run_ai_suite.sh --test "mlx,pytorch"
```

### 1. Upload Scripts to your PVC
Copy the `benchmarks/` folder (which will then act as your `benchmark_scripts` directory) to your Ceph server so that it sits at `/data/shared/benchmark_scripts/` (or whichever path your NFS server exports).

### 2. Run the AI Benchmark Suite
Apply the Kubernetes Job. Make sure the NFS mount paths inside the YAML match your environment.

```bash
kubectl apply -f ai_test_suite.yaml
```

*(Optional)* You can modify the `command` arguments in `ai_test_suite.yaml` to include `--test "coreml, llm"` run flags if you only want to schedule a subset of the benchmarks on your cluster. Valid keywords: `cpu`, `mlx`, `coreml`, `pytorch`, `llm`, `nfs`, and `mlx_bench`.

Once the pod starts, follow the logs. It will take a few minutes to install `coremltools` inside the virtual environment before execution starts.
```bash
kubectl logs job/macos-ai-benchmark -f
```

*(Note: Apply the exact same YAML to your `macOS-vz-kubelet` cluster. The CoreML test is designed to not crash inside the VM, allowing you to directly compare the millisecond inference times between the two platforms.)*

### 3. Run the Storage / Network Overhead Benchmarks

**On the macOS Node:**
```bash
kubectl apply -f nfs_io_mac.yaml
kubectl logs job/macos-fio-benchmark -f
```

**On a standard Linux Node (Baseline):**
```bash
# Verify the namespace and PVC name in the YAML match your Ceph environment first!
kubectl apply -f ceph_io_linux.yaml
kubectl logs job/linux-fio-benchmark -f
```
Compare the throughput (`bw`) and IOPS (`iops`) between the macOS NFS sidecar approach and standard Linux CSI drivers to calculate the exact NFS performance penalty.
