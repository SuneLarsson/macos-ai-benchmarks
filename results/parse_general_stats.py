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
    
    master_data = []
    header = ["Environment", "Benchmark_Name", "Iteration", "Iteration_Time_S"]
    
    for file_path in json_files:
        basename = os.path.basename(file_path)
        if basename.startswith("llm_stats_"):
            continue # Let parse_llm_stats.py handle these
            
        env_name = os.path.basename(os.path.dirname(file_path))
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                # Look for the raw_times_s array which contains every iteration
                if isinstance(data, dict) and "raw_times_s" in data:
                    benchmark_name = data.get("stats", {}).get("benchmark", basename)
                    raw_times = data["raw_times_s"]
                    
                    for idx, t in enumerate(raw_times):
                        master_data.append([
                            env_name, benchmark_name, idx + 1, t
                        ])
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
                
    # Sort by Benchmark, Iteration, Environment
    master_data.sort(key=lambda x: (x[1], x[2], x[0]))
    
    out_file = os.path.join(script_dir, "combined_general_scatter_data.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(master_data)
        
    print(f"Created General benchmark data file for scatter plots: {out_file}")

if __name__ == "__main__":
    main()
