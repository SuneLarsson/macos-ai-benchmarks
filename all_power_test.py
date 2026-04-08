import multiprocessing
import time

from cpu_ml_baseline import benchmark_cpu
from gpu_mlx_test import benchmark_gpu
from npu_coreml_test import benchmark_npu

def run_cpu():
    benchmark_cpu(prefix="allpower")

def run_gpu():
    benchmark_gpu(prefix="allpower")

def run_npu():
    benchmark_npu(prefix="allpower")

def benchmark_all():
    print("=== Starting ALL POWER TEST ===")
    print("Stressing CPU, GPU, and NPU concurrently...")
    
    start_time = time.time()
    
    p1 = multiprocessing.Process(target=run_cpu)
    p2 = multiprocessing.Process(target=run_gpu)
    p3 = multiprocessing.Process(target=run_npu)
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()
        
    end_time = time.time()
    print(f"=== All Power Test Completed in {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    benchmark_all()
