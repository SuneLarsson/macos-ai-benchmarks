import os
import json
import glob
import argparse
import statistics
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='', help='Optional date string to filter by')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We want to find all json files
    file_pattern = f"llm_stats_*{args.date}*.json" if args.date else "llm_stats_*.json"
    search_pattern = os.path.join(script_dir, "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print("No llm_stats_*.json files found.")
        return
        
    # Dictionary structure: data[model_name][environment] = {'power': [], 'tpw': [], 'tps': []}
    data = defaultdict(lambda: defaultdict(lambda: {'power': [], 'tpw': [], 'tps': []}))
    
    for file_path in json_files:
        # Ignore any files inside a directory named 'old'
        if "old" in file_path.lower().split(os.sep):
            continue
            
        filename = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        # Determine environment
        env_name = "Unknown"
        if "exec" in dir_name.lower():
            env_name = "Mac"
        elif "linux" in dir_name.lower():
            if "_OPTIMIZED" in filename:
                env_name = "H100 16-bit (Compiled)"
            elif "_NATIVE" in filename:
                env_name = "H100 16-bit (Native)"
            else:
                env_name = "H100 4-bit (Direct)"
        else:
            continue # Skip unknown environments like 'vm' unless requested
            
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if isinstance(content, dict) and "runs" in content:
                    # Clean up model name
                    model_name = content.get("model", filename)
                    model_name = model_name.split("/")[-1] # Remove prefix
                    model_name = model_name.replace("-Instruct", "").replace("-4bit", "").replace("_OPTIMIZED.json", "").replace("_NATIVE.json", "").replace(".json", "")
                    if model_name.startswith("llm_stats_"):
                        model_name = model_name.replace("llm_stats_", "")
                    
                    runs = content["runs"]
                    for run in runs:
                        power = run.get("average_power_w", None)
                        tpw = run.get("tokens_per_watt", None)
                        tps = run.get("tokens_per_second", None)
                        
                        if power is not None and tpw is not None and tps is not None:
                            data[model_name][env_name]['power'].append(float(power))
                            data[model_name][env_name]['tpw'].append(float(tpw))
                            data[model_name][env_name]['tps'].append(float(tps))
                            
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")

    if not data:
        print("No valid power metrics found in the parsed files.")
        return

    # Generate LaTeX Table
    out_file = os.path.join(script_dir, "llm_power_metrics_table.tex")
    
    # Define the order we want to display environments
    env_order = ["Mac", "H100 4-bit (Direct)", "H100 16-bit (Native)", "H100 16-bit (Compiled)"]
    
    models = sorted(data.keys())
    
    with open(out_file, 'w') as f:
        f.write("% Add to your preamble: \\usepackage{multirow}\n")
        f.write("\\begin{table}[hbt!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Environment} & \\textbf{Avg TPS} & \\textbf{Avg Power (W)} & \\textbf{Avg TPW} \\\\\n")
        f.write("\\hline\n")
        
        for model in models:
            envs_present = [env for env in env_order if env in data[model] and len(data[model][env]['power']) > 0]
            if not envs_present:
                continue
                
            num_envs = len(envs_present)
            safe_model = model.replace('_', '\\_')
            
            for i, env in enumerate(envs_present):
                power_vals = data[model][env]['power']
                tpw_vals = data[model][env]['tpw']
                tps_vals = data[model][env]['tps']
                
                if len(power_vals) >= 1:
                    tps_mean = f"{statistics.mean(tps_vals):.2f}"
                    p_mean = f"{statistics.mean(power_vals):.2f}"
                    t_mean = f"{statistics.mean(tpw_vals):.4f}"
                else:
                    tps_mean = "-"
                    p_mean = "-"
                    t_mean = "-"
                    
                if i == 0:
                    model_col = f"\\multirow{{{num_envs}}}{{*}}{{{safe_model}}}"
                else:
                    model_col = ""
                    
                f.write(f"{model_col} & {env} & {tps_mean} & {p_mean} & {t_mean} \\\\\n")
                
            f.write("\\hline\n")
            
        f.write("\\end{tabular}\n")
        f.write("\\caption{LLM Efficiency Comparison: Mac vs H100 (4-bit, Native, Compiled)}\n")
        f.write("\\label{tab:llm_power_comparison}\n")
        f.write("\\end{table}\n")
        
    print(f"Created LaTeX Power Metrics Table: {out_file}")

if __name__ == "__main__":
    main()
