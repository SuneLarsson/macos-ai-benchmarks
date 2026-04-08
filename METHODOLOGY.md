# Benchmarking Methodology & Framework Justification

This document outlines the software frameworks used throughout the `macos-exec-kubelet` AI benchmarking suite. It details what each framework measures, how it is implemented, and the academic or technical motivation for its inclusion in evaluating Apple Silicon against bare-metal Linux workers under Kubernetes orchestration.

## 1. MLX and MLX-LM (`mlx`, `mlx-lm`)

**What it tests:** Native GPU compute, unified memory bandwidth, and 4-bit quantized Large Language Model (LLM) inference.

**How it is used:** 
- **Raw Throughput:** `gpu_mlx_test.py` generates large (4096x4096) uniform matrices and repeatedly multiplies them to saturate the GPU cores.
- **Inference Metrics:** `llm_inference_bench.py` and `nfs_data_bench.py` use `mlx-lm` to load heavily quantized (4-bit) instruction-tuned language models (e.g., Llama-3.2, Mistral, Qwen) to measure Time-To-First-Token (TTFT) and Tokens-Per-Second (TPS).

**Why it was chosen:**
Originally designed by Apple machine learning researchers, MLX is a NumPy-like array framework tailored to leverage the unique unified memory architecture of Apple Silicon. While traditional machine learning frameworks require expensive memory copies between the CPU RAM and discrete GPU VRAM, MLX executes directly on the shared memory address space. 

Furthermore, using `mlx-lm` with 4-bit quantization allows models as large as 7 Billion parameters (which natively require ~14GB of RAM in 16-bit precision) to run deep inside the 16GB limit of a consumer Mac Mini M4 without triggering swap degradation. This guarantees that the benchmark measures pure silicon performance instead of operating system pagefile thrashing. Because MLX recently introduced native CUDA support, it also perfectly facilitates cross-platform evaluations on Linux hardware.

## 2. CoreML Tools (`coremltools`)

**What it tests:** Hardware-accelerated neural network execution explicitly targeting the Apple Neural Engine (ANE).

**How it is used:** `coreml_hardware_bench.py` dynamically compiles a massive multi-layer perceptron (MLP) containing heavy inner-product layers into an `.mlpackage`. It structurally evaluates hardware inference latency natively by looping through all three explicit compute targets sequentially: `ct.ComputeUnit.CPU_ONLY`, `ct.ComputeUnit.CPU_AND_GPU`, and `ct.ComputeUnit.ALL`.

**Why it was chosen:**
Apple Silicon contains a dedicated Neural Engine coprocessor engineered specifically for deterministic, low-power machine learning inference operations. The only officially supported pathway to harness the ANE programmatically in Python is by compiling neural networks down to the CoreML model format. By explicitly stepping the identical model through `CPU_ONLY`, `CPU_AND_GPU`, and `ALL` computations natively inside the Python script, we generate three overlapping JSON output distributions. This mathematically proves and quantifies the exact throughput acceleration advantage provided by the GPU and the Neural Engine over standard CoreML processing.

## 3. PyTorch (`torch`, `transformers`)

**What it tests:** Hardware-agnostic cross-platform tensor operations and machine learning layers.

**How it is used:** `cross_platform_ai_bench.py` establishes identical code logic that conditionally connects to Apple's Metal Performance Shaders (`mps`) or NVIDIA's `cuda` backend, depending entirely on the host operating system discovering the payload.

**Why it was chosen:**
PyTorch is the undisputed industry standard for deep learning research and production. In the context of a distributed Kubernetes environment containing both `macos-exec-kubelet` runners and standard Linux nodes, utilizing PyTorch establishes a perfectly controlled variable. By keeping the Python script identical, any disparity in execution time between the Linux baseline job and the macOS job is exclusively attributed to the underlying hardware architecture and the virtualized execution layer, creating an unbreakable 1-to-1 comparison metric.

## 4. MLX Benchmark Suite (`TristanBilot/mlx-benchmark`)

**What it tests:** Micro-profiling of discrete MLX mathematical operations (convolutions, activations, element-wise math).

**How it is used:** The entire repository is automatically pulled and executed by the deployment logic during job execution to dump a massive profiling matrix.

**Why it was chosen:**
While our custom benchmarking scripts measure "macro" workloads (predictive inference, full matrix loads, and data pipeline bottlenecks), the official `mlx-benchmark` repository computationally isolates "micro" workloads. It measures the fundamental operational latency of the framework itself. Gathering this exact data natively maps our specific networking and virtualization infrastructure to the broader academic literature analyzing Apple Silicon AI performance.

## 5. Flexible I/O Tester (`fio`)

**What it tests:** Persistent volume storage constraints, specifically network-attached storage (NFS vs Ceph CSI).

**How it is used:** Executed within separate Kubernetes Jobs (`nfs_io_mac.yaml`, `ceph_io_linux.yaml`) to mimic the sequential read throughput of loading a gigabyte model checkpoint into memory, as well as the random-access throughput required to shuffle dataset batches.

**Why it was chosen:**
In modern AI infrastructure, compute is rarely the only bottleneck; input/output (I/O) saturation frequently dictates maximum throughput, especially when loading massive multi-gigabyte models across clustered persistent volumes. Since the `macos-exec-kubelet` leverages a Linux sidecar to proxy Ceph storage over an NFS network socket, `fio` provides the most rigorous possible mechanism to quantify the virtualization overhead penalty of that specific network hop compared to native Linux Ceph drivers.

## 6. NumPy (`numpy`)

**What it tests:** Single and multi-threaded CPU floating-point math processing.

**How it is used:** `cpu_ml_baseline.py` measures traditional mathematical procedures natively on the CPU threads without leveraging explicit hardware ML accelerators like Metal or CUDA.

**Why it was chosen:**
It provides a stable, omnipresent baseline. Since it bypasses AI specific coprocessors (NPU/GPU), it isolates the raw performance of the CPU instruction set. This serves as a control variable for validating how much latency our AI-accelerated frameworks are actually saving natively on the node.
