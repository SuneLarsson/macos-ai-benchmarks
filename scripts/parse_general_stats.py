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
    
    # We want to match all JSON files, but we'll exclude the LLM ones
    file_pattern = f"*{args.date}*.json" if args.date else "*.json"
    search_pattern = os.path.join(script_dir, "**", file_pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    benchmarks = {} # format: { benchmark_name: { env_name: stats_dict } }
    
    for file_path in json_files:
        basename = os.path.basename(file_path)
        if basename.startswith("llm_stats_"):
            continue 
            
        env_name = os.path.basename(os.path.dirname(file_path))
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if "stats" in data:
                    stats = data["stats"]
                    benchmark_name = stats.get("benchmark", basename)
                    
                    if benchmark_name not in benchmarks:
                        benchmarks[benchmark_name] = {}
                    benchmarks[benchmark_name][env_name] = stats
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
                
    # Identify all environments to determine baseline
    all_envs = set()
    for env_data in benchmarks.values():
        all_envs.update(env_data.keys())
    
    all_envs = sorted(list(all_envs))
    if not all_envs:
        print("No stats data found.")
        return
        
    base_env = all_envs[0]
    compare_envs = all_envs[1:]
    
    metrics = ["mean_s", "p50_s", "p90_s", "p95_s", "p99_s"]
    
    header = ["Benchmark", "Metric", base_env]
    for c_env in compare_envs:
        header.append(c_env)
        header.append(f"{c_env}_vs_{base_env}_%_diff")
        
    out_data = []
    
    # Sort benchmarks alphabetically
    for b_name in sorted(benchmarks.keys()):
        env_data = benchmarks[b_name]
        
        for m in metrics:
            base_val = env_data.get(base_env, {}).get(m, None)
            
            # We'll skip metrics that don't exist
            if base_val is None and all(env_data.get(c, {}).get(m, None) is None for c in compare_envs):
                continue
                
            row = [b_name, m]
            row.append(base_val if base_val is not None else "N/A")
            
            for c_env in compare_envs:
                comp_val = env_data.get(c_env, {}).get(m, None)
                row.append(comp_val if comp_val is not None else "N/A")
                
                diff_str = "N/A"
                if base_val is not None and comp_val is not None and base_val != 0:
                    diff_raw = ((comp_val - base_val) / base_val) * 100
                    prefix = "+" if diff_raw > 0 else ""
                    diff_str = f"{prefix}{diff_raw:.2f}%"
                    
                row.append(diff_str)
                
            out_data.append(row)
            
    out_file = os.path.join(script_dir, "combined_general_stats_diff.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_data)
        
    print(f"Created General stats difference file: {out_file}")

if __name__ == "__main__":
    main()
