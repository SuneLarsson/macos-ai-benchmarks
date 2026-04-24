import time
import os
import json
import argparse
import signal
import sys
import subprocess
import threading
import re
import mlx.core as mx
from mlx_lm import load
try:
    # In newer versions of mlx_lm, generate_step was moved to mlx_lm.generate
    from mlx_lm.generate import generate_step
except ImportError:
    # Fallback for older versions of mlx_lm
    from mlx_lm.utils import generate_step

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)

class PowerTracker:
    def __init__(self):
        self.process = None
        self.power_samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _read_output(self):
        sudo_password = os.environ.get("SUDO_PASSWORD")
        cmd = ["sudo"]
        if sudo_password:
            cmd.extend(["-S"])
        else:
            cmd.extend(["-n"])
            
        cmd.extend(["powermetrics", "-i", "100", "--samplers", "cpu_power,gpu_power,ane_power"])
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            if sudo_password and self.process.stdin:
                self.process.stdin.write(sudo_password + "\n")
                self.process.stdin.flush()
                
            combined_pattern = re.compile(r"Combined Power \(.*?\):\s+(\d+)\s+mW")
            
            while not self._stop_event.is_set():
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                
                m = combined_pattern.search(line)
                if m:
                    self.power_samples.append(float(m.group(1)))
                    
        except Exception as e:
            print(f"PowerTracker error: {e}")
            
    def start(self):
        self.power_samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_output)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                
        if self._thread:
            self._thread.join(timeout=2)
            
        if self.power_samples:
            avg_mw = sum(self.power_samples) / len(self.power_samples)
            return avg_mw / 1000.0  # Return Watts
        else:
            if self.process and self.process.stderr:
                err = self.process.stderr.read().strip()
                if err:
                    print(f"[PowerTracker] No samples collected! Error: {err}")
                else:
                    print(f"[PowerTracker] No samples collected, but no error was output from powermetrics. Regex may have failed.")
        return None

# The exact four models explicitly cited in "Evaluating small quantized language models on apple silicon"
# We utilize the 4-bit mlx-community quantized variants to ensure they fit in the 16GB Mac Mini unified memory,
# explicitly mimicking the paper's methodology.
MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
]

def benchmark_inference(iterations=1, seed=42, prefix="", output_dir="results"):
    print(f"Starting MLX-LM Inference Benchmark (Paper Replication) with base seed {seed}...")
    
    prompt = "Write a comprehensive 500 word essay on the history of computers:"
    
    for model_id in MODELS:
        print(f"\n=====================================")
        print(f"Evaluating: {model_id}")
        
        start_load = time.perf_counter()
        try:
            model, tokenizer = load(model_id)
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue
            
        load_time = time.perf_counter() - start_load
        print(f"Model loaded securely in {load_time:.2f}s.")
        
        # Format prompt using the model's chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_prompt = prompt
            
        prompt_tokens = mx.array(tokenizer.encode(text_prompt))
        
        # Hardware Warmup
        print("Starting hardware warmup...")
        for _, _ in zip(range(2), generate_step(prompt_tokens, model)):
            pass
        mx.eval(model.parameters()) # ensure synchronization
        
        power_tracker = PowerTracker()
        all_stats = []
        for i in range(iterations):
            current_seed = seed + i
            mx.random.seed(current_seed)
            print(f"Executing benchmark generation {i+1}/{iterations} (seed {current_seed}, max 256 tokens)...")
            
            first_token_time = None
            tokens_generated = 0
            
            power_tracker.start()
            start_gen = time.perf_counter()
            
            # Iterate generation step natively to accurately timestamp First Token
            for token, _ in generate_step(prompt_tokens, model):
                # mx.eval forces the async graph execution to resolve so we get exact timing
                mx.eval(token) 
                
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_gen
                    
                tokens_generated += 1
                if tokens_generated >= 256:
                    break
                    
            total_time = time.perf_counter() - start_gen
            avg_power_w = power_tracker.stop()
            
            tps = tokens_generated / total_time
            ttft = first_token_time if first_token_time else total_time
            
            stats = {
                "iteration": i + 1,
                "seed": current_seed,
                "time_to_first_token_s": float(ttft),
                "total_generation_time_s": float(total_time),
                "tokens_generated": int(tokens_generated),
                "tokens_per_second": float(tps)
            }
            if avg_power_w is not None:
                stats["average_power_w"] = float(avg_power_w)
                stats["tokens_per_watt"] = float(tps / avg_power_w) if avg_power_w > 0 else 0.0
                
            all_stats.append(stats)
            
            print(f"--- Iteration {i+1} Results ---")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
                    
        os.makedirs(output_dir, exist_ok=True)
        filename_prefix = f"{prefix}_" if prefix else ""
        safe_model_name = model_id.split("/")[-1]
        filename = f"{output_dir}/{filename_prefix}llm_stats_{safe_model_name}.json"
        
        with open(filename, "w") as f:
            json.dump({
                "benchmark": "LLM Inference (MLX-LM)",
                "model": model_id,
                "load_time_s": float(load_time),
                "target_iterations": iterations,
                "runs": all_stats
            }, f, indent=4)
            
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='Number of times to run the evaluation loop')
    parser.add_argument('--timeout', type=int, default=0, help='Timeout in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    target_runs = max(1, args.runs // 5)
    print(f"LLM Bench: Adjusted target runs from {args.runs} to {target_runs}")
    benchmark_inference(iterations=target_runs, seed=args.seed, output_dir=args.output_dir)
