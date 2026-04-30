import time
import os
import json
import argparse
import signal
import sys
import subprocess
import threading
import re
import shutil

def timeout_handler(signum, frame):
    print("Timeout reached! Exiting.")
    sys.exit(124)

HAS_MLX = False
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    try:
        from mlx_lm.generate import generate_step as mlx_generate_step
    except ImportError:
        from mlx_lm.utils import generate_step as mlx_generate_step
    HAS_MLX = True
except Exception as e:
    print(f"MLX or mlx-lm failed to initialize (falling back if CUDA is available): {e}")

HAS_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        HAS_CUDA = True
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    pass

class PowerTracker:
    def __init__(self):
        self.process = None
        self.power_samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self.is_nvidia = sys.platform == "linux" and shutil.which("nvidia-smi") is not None

    def _read_output(self):
        if self.is_nvidia:
            cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "--loop-ms=100"]
            try:
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
                )
                while not self._stop_event.is_set():
                    line = self.process.stdout.readline()
                    if not line and self.process.poll() is not None:
                        break
                    if line.strip():
                        try:
                            # e.g., "15.23"
                            self.power_samples.append(float(line.strip()))
                        except ValueError:
                            pass
            except Exception as e:
                print(f"PowerTracker (nvidia-smi) error: {e}")
        else:
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
                print(f"PowerTracker (powermetrics) error: {e}")
            
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
            avg = sum(self.power_samples) / len(self.power_samples)
            if self.is_nvidia:
                return avg # nvidia-smi gives watts directly
            else:
                return avg / 1000.0  # powermetrics gives mW
        else:
            if self.process and self.process.stderr:
                err = self.process.stderr.read().strip()
                if err:
                    print(f"[PowerTracker] No samples collected! Error: {err}")
                else:
                    print(f"[PowerTracker] No samples collected, but no error was output. Regex may have failed.")
        return None

if HAS_CUDA:
    class TTFTTracker(StoppingCriteria):
        def __init__(self):
            self.start_time = None
            self.ttft = None
            self.tokens_generated = 0
            
        def reset(self):
            self.start_time = time.perf_counter()
            self.ttft = None
            self.tokens_generated = 0
            
        def __call__(self, input_ids, scores, **kwargs):
            if self.ttft is None:
                self.ttft = time.perf_counter() - self.start_time
            self.tokens_generated += 1
            return False

MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
]

CUDA_MODELS = {
    "mlx-community/Llama-3.2-1B-Instruct-4bit": "unsloth/Llama-3.2-1B-Instruct",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "Qwen/Qwen2.5-1.5B-Instruct",
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "unsloth/Llama-3.2-3B-Instruct",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit": "mistralai/Mistral-7B-Instruct-v0.3"
}

def benchmark_inference(iterations=1, seed=42, tokens=256, prefix="", output_dir="results"):
    use_cuda = HAS_CUDA and not HAS_MLX
    engine_name = "Transformers/CUDA" if use_cuda else "MLX-LM/Apple Silicon"
    print(f"Starting LLM Inference Benchmark ({engine_name}) with base seed {seed}...")
    
    prompt = "Write a comprehensive 500 word essay on the history of computers:"
    
    for model_id in MODELS:
        print(f"\n=====================================")
        
        actual_model_id = CUDA_MODELS.get(model_id, model_id) if use_cuda else model_id
        print(f"Evaluating: {actual_model_id} (Requested: {model_id})")
        
        start_load = time.perf_counter()
        try:
            if use_cuda:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(actual_model_id)
                model = AutoModelForCausalLM.from_pretrained(actual_model_id, quantization_config=quantization_config, device_map="auto")
            else:
                model, tokenizer = mlx_load(model_id)
        except Exception as e:
            print(f"Failed to load {actual_model_id}: {e}")
            continue
            
        load_time = time.perf_counter() - start_load
        print(f"Model loaded securely in {load_time:.2f}s.")
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_prompt = prompt
            
        print("Starting hardware warmup...")
        if use_cuda:
            model_inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                dummy_tracker = TTFTTracker()
                dummy_stopping_criteria = StoppingCriteriaList([dummy_tracker])
                _ = model.generate(
                    **model_inputs,
                    min_new_tokens=10,
                    max_new_tokens=10,
                    do_sample=True,
                    stopping_criteria=dummy_stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
        else:
            prompt_tokens = mx.array(tokenizer.encode(text_prompt))
            for _, _ in zip(range(2), mlx_generate_step(prompt_tokens, model)):
                pass
            mx.eval(model.parameters())
            
        power_tracker = PowerTracker()
        all_stats = []
        
        for i in range(iterations):
            current_seed = seed + i
            print(f"Executing benchmark generation {i+1}/{iterations} (seed {current_seed}, exactly {tokens} tokens)...")
            
            first_token_time = None
            tokens_generated = 0
            
            power_tracker.start()
            
            if use_cuda:
                torch.manual_seed(current_seed)
                ttft_tracker = TTFTTracker()
                stopping_criteria = StoppingCriteriaList([ttft_tracker])
                
                ttft_tracker.reset()
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **model_inputs,
                        min_new_tokens=tokens,
                        max_new_tokens=tokens,
                        do_sample=True,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                torch.cuda.synchronize()
                total_time = time.perf_counter() - ttft_tracker.start_time
                
                tokens_generated = ttft_tracker.tokens_generated
                if tokens_generated == 0:
                    tokens_generated = output_ids.shape[1] - model_inputs["input_ids"].shape[1]
                first_token_time = ttft_tracker.ttft
            else:
                mx.random.seed(current_seed)
                start_gen = time.perf_counter()
                
                try:
                    gen_iterator = mlx_generate_step(prompt_tokens, model, max_tokens=tokens)
                except TypeError:
                    # older mlx-lm versions might not accept max_tokens in generate_step
                    gen_iterator = mlx_generate_step(prompt_tokens, model)
                    
                for token, _ in gen_iterator:
                    mx.eval(token)
                    
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_gen
                        
                    tokens_generated += 1
                    if tokens_generated >= tokens:
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
                "benchmark": f"LLM Inference ({engine_name})",
                "model": actual_model_id,
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
    parser.add_argument('--tokens', type=int, default=256, help='Number of tokens to generate')
    args = parser.parse_args()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    target_runs = max(1, args.runs // 5)
    print(f"LLM Bench: Adjusted target runs from {args.runs} to {target_runs}")
    benchmark_inference(iterations=target_runs, seed=args.seed, tokens=args.tokens, output_dir=args.output_dir)
