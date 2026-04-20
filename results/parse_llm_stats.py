import os
import json
import glob
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='', help='Optional date string to filter by (e.g., 2026-04-20)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_pattern = f"llm_stats_*{args.date}*.json" if args.date else "llm_stats_*.json"
    search_pattern = os.path.join(script_dir, "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print("No llm_stats_*.json files found.")
        return
        
    master_data = []
    # These are the columns needed for scatter plots
    header = [
        "Environment", "Model", "Iteration", 
        "Tokens_Per_Second", "Time_To_First_Token_S", 
        "Generation_Time_S", "Prompt_Tokens", "Generation_Tokens"
    ]
    
    for file_path in json_files:
        env_name = os.path.basename(os.path.dirname(file_path)) # e.g. "vm" or "exec"
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "runs" in data:
                    # Clean up model name
                    model_name = data.get("model", os.path.basename(file_path))
                    model_name = model_name.split("/")[-1] # Remove mlx-community/ prefix if it exists
                    runs = data["runs"]
                    
                    for idx, run in enumerate(runs):
                        iteration = run.get("iteration", idx + 1)
                        tps = run.get("tokens_per_second", "")
                        ttft = run.get("time_to_first_token_s", "")
                        gen_time = run.get("generation_time_s", "")
                        prompt_tok = run.get("prompt_tokens", "")
                        gen_tok = run.get("generation_tokens", "")
                        
                        master_data.append([
                            env_name, model_name, iteration,
                            tps, ttft, gen_time, prompt_tok, gen_tok
                        ])
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
                
    # Sort to neatly align iterations
    master_data.sort(key=lambda x: (x[1], x[2], x[0]))
    
    out_file = os.path.join(script_dir, "combined_llm_scatter_data.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(master_data)
        
    print(f"Created LLM data file for scatter plots: {out_file}")

if __name__ == "__main__":
    main()
