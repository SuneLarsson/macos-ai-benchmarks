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
        "Generation_Time_S", "Prompt_Tokens", "Generation_Tokens",
        "Average_Power_W", "Tokens_Per_Watt"
    ]
    
    # For LaTeX preprocessing
    import statistics
    from collections import defaultdict
    agg_data = defaultdict(lambda: defaultdict(list))
    
    for file_path in json_files:
        # Ignore any files inside a directory named 'old'
        if "old" in file_path.lower().split(os.sep):
            continue
            
        env_name = os.path.basename(os.path.dirname(file_path)) # e.g. "vm" or "exec"
        if env_name.startswith("results_"):
            env_name = env_name.replace("results_", "")
        if env_name not in ['exec', 'vm', 'linux']:
            continue
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "runs" in data:
                    # Clean up model name
                    model_name = data.get("model", os.path.basename(file_path))
                    model_name = model_name.split("/")[-1] # Remove mlx-community/ prefix if it exists
                    model_name = model_name.replace("-Instruct", "").replace("-4bit", "")
                    runs = data["runs"]
                    
                    for idx, run in enumerate(runs):
                        iteration = run.get("iteration", idx + 1)
                        tps = run.get("tokens_per_second", "")
                        ttft = run.get("time_to_first_token_s", "")
                        gen_time = run.get("total_generation_time_s", run.get("generation_time_s", ""))
                        prompt_tok = run.get("prompt_tokens", "")
                        gen_tok = run.get("tokens_generated", run.get("generation_tokens", ""))
                        
                        power = run.get("average_power_w", "")
                        tpw = run.get("tokens_per_watt", "")
                        
                        master_data.append([
                            env_name, model_name, iteration,
                            tps, ttft, gen_time, prompt_tok, gen_tok,
                            power, tpw
                        ])
                        
                        if tps != "": agg_data[(env_name, model_name)]["tps"].append(float(tps))
                        if ttft != "": agg_data[(env_name, model_name)]["ttft"].append(float(ttft))
                        if gen_time != "": agg_data[(env_name, model_name)]["gen_time"].append(float(gen_time))
                        if power != "": agg_data[(env_name, model_name)]["power"].append(float(power))
                        if tpw != "": agg_data[(env_name, model_name)]["tpw"].append(float(tpw))
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

    # --- Generate PGFPlots LaTeX Snippets for Scatter Plots ---
    # Group coordinates by Model and Environment
    coords_by_model_env = defaultdict(lambda: defaultdict(list))
    for row in master_data:
        env_name, model_name = row[0], row[1]
        tps, ttft, gen_time, power, tpw = row[3], row[4], row[5], row[8], row[9]
        coords_by_model_env[model_name][env_name].append({
            'tps': tps, 'ttft': ttft, 'gen_time': gen_time, 'power': power, 'tpw': tpw
        })

    def write_preamble(f):
        f.write("% Add to your preamble:\n")
        f.write("% \\usepackage{tikz}\n")
        f.write("% \\usepackage{pgfplots}\n")
        f.write("% \\pgfplotsset{compat=1.18}\n\n")

    models = sorted(coords_by_model_env.keys())
    
    env_styles = {
        'exec': {'label': 'User-space', 'color': 'blue!60', 'mark': '*'},
        'vm': {'label': 'Virtualization', 'color': 'red!60', 'mark': 'triangle*'},
        'linux': {'label': 'Linux', 'color': 'green!60', 'mark': 'square*'}
    }

    # 1. Individual Performance Plots
    perf_ind_file = os.path.join(script_dir, "llm_latex_performance_individual.tex")
    with open(perf_ind_file, 'w') as f:
        write_preamble(f)
        for model in models:
            envs = coords_by_model_env[model]
            safe_model = model.replace('_', '\\_')
            
            f.write(f"% --- Scatter Plot for {model}: TTFT vs TPS ---\n")
            f.write("\\begin{figure}[hbt!]\n")
            f.write("\\centering\n")
            f.write("\\begin{tikzpicture}\n")
            f.write("\\begin{axis}[\n")
            f.write("    width=\\textwidth,\n")
            f.write("    height=8cm,\n")
            f.write(f"    title={{{safe_model} - Performance}},\n")
            f.write("    xlabel={Time to First Token (s)},\n")
            f.write("    ylabel={Tokens Per Second},\n")
            f.write("    legend style={at={(0.5,-0.2)},anchor=north,legend columns=-1,draw=none},\n")
            f.write("    grid=major,\n")
            f.write("    scaled x ticks=false,\n")
            f.write("    x tick label style={/pgf/number format/fixed, /pgf/number format/precision=3},\n")
            f.write("    ymin=0,\n")
            f.write("]\n")
            
            for env in ['exec', 'vm', 'linux']:
                if env not in envs: continue
                coords = [f"({pt['ttft']},{pt['tps']})" for pt in envs[env] if pt['ttft'] != "" and pt['tps'] != ""]
                if coords:
                    style = env_styles[env]
                    f.write(f"\\addplot[\n    only marks,\n    mark size=3pt,\n    mark={style['mark']},\n    color={style['color']}\n] coordinates {{\n    {' '.join(coords)}\n}};\n")
                    f.write(f"\\addlegendentry{{{style['label']}}}\n")
            
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
            f.write(f"\\caption{{Time to First Token vs Tokens Per Second for {safe_model}}}\n")
            f.write("\\end{figure}\n\n")

    # 2. Efficiency Table (Power Util)
    eff_table_file = os.path.join(script_dir, "llm_latex_efficiency_table.tex")
    with open(eff_table_file, 'w') as f:
        f.write("% Add to your preamble: \\usepackage{multirow}\n")
        f.write("\\begin{table}[hbt!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Environment} & \\textbf{Power Mean (W)} & \\textbf{TPW Mean} & \\textbf{TPW Std} \\\\\n")
        f.write("\\hline\n")
        
        for model in models:
            envs = coords_by_model_env[model]
            safe_model = model.replace('_', '\\_')
            
            # Find which envs actually have power data
            valid_envs = []
            for env in ['exec', 'vm', 'linux']:
                if env in envs:
                    pts = envs[env]
                    power_vals = [float(pt['power']) for pt in pts if pt['power'] != ""]
                    tpw_vals = [float(pt['tpw']) for pt in pts if pt['tpw'] != ""]
                    if power_vals and tpw_vals:
                        valid_envs.append((env, power_vals, tpw_vals))
                        
            if not valid_envs:
                continue
                
            num_envs = len(valid_envs)
            
            for i, (env, power_vals, tpw_vals) in enumerate(valid_envs):
                if len(power_vals) > 1:
                    p_mean = f"{statistics.mean(power_vals):.4f}"
                    p_std = f"{statistics.stdev(power_vals):.4f}"
                    t_mean = f"{statistics.mean(tpw_vals):.4f}"
                    t_std = f"{statistics.stdev(tpw_vals):.4f}"
                else:
                    p_mean = f"{power_vals[0]:.4f}"
                    p_std = "-"
                    t_mean = f"{tpw_vals[0]:.4f}"
                    t_std = "-"
                    
                env_label = env_styles[env]['label']
                
                if i == 0:
                    model_col = f"\\multirow{{{num_envs}}}{{*}}{{{safe_model}}}"
                else:
                    model_col = ""
                    
                f.write(f"{model_col} & {env_label} & {p_mean} & {t_mean} & {t_std} \\\\\n")
                
            f.write("\\hline\n")
            
        f.write("\\end{tabular}\n")
        f.write("\\caption{LLM Efficiency Comparison: Average Power and Tokens per Watt}\n")
        f.write("\\label{tab:llm_efficiency}\n")
        f.write("\\end{table}\n\n")

    # 3. Composite Plots setup
    markers = ['*', 'triangle*', 'square*', 'diamond*', 'pentagon*', 'x', '+']
    model_marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}
    env_color_map = {'exec': 'blue', 'vm': 'red', 'linux': 'green'}
    env_label_map = {'exec': 'User-space', 'vm': 'Virtualization', 'linux': 'Linux'}
    
    # 4. Composite Performance Plot
    perf_comp_file = os.path.join(script_dir, "llm_latex_performance_composite.tex")
    with open(perf_comp_file, 'w') as f:
        write_preamble(f)
        f.write("\\begin{figure}[hbt!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("    width=\\textwidth,\n")
        f.write("    height=8cm,\n")
        f.write("    title={LLM Performance Comparison (All Models)},\n")
        f.write("    xlabel={Time to First Token (s)},\n")
        f.write("    ylabel={Tokens Per Second},\n")
        f.write("    legend pos=north east,\n")
        f.write("    grid=major,\n")
        f.write("    scaled x ticks=false,\n")
        f.write("    x tick label style={/pgf/number format/fixed, /pgf/number format/precision=3},\n")
        f.write("    xmin=0, ymin=0,\n")
        f.write("]\n")
        
        # Write custom decoupled legend entries for Models only
        for model in models:
            safe_model = model.replace('_', '\\_')
            f.write(f"\\addlegendimage{{only marks, mark={model_marker_map[model]}, color=darkgray}}\n")
            f.write(f"\\addlegendentry{{{safe_model}}}\n")
        
        for model in models:
            envs = coords_by_model_env[model]
            for env in ['exec', 'vm', 'linux']:
                if env not in envs: continue
                coords = [f"({pt['ttft']},{pt['tps']})" for pt in envs[env] if pt['ttft'] != "" and pt['tps'] != ""]
                if coords:
                    f.write(f"\\addplot[\n    only marks,\n    mark size=3pt,\n    mark={model_marker_map[model]},\n    color={env_color_map[env]}!60\n] coordinates {{\n    {' '.join(coords)}\n}};\n")
        
        f.write("\\end{axis}\n\\end{tikzpicture}\n")
        f.write("\\caption{Composite Time to First Token vs Tokens Per Second. Environments: User-space (Blue), Virtualization (Red).}\n")
        f.write("\\end{figure}\n\n")

    # 5. Simple Text Report for Mean TTFT and TPS
    means_report_file = os.path.join(script_dir, "llm_performance_means.csv")
    with open(means_report_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Environment", "Mean_TTFT_S", "StdDev_TTFT_S", "Diff_TTFT_%", "Mean_TPS", "StdDev_TPS", "Diff_TPS_%"])
        
        for model in models:
            envs = coords_by_model_env[model]
            
            # Find baseline 'exec'
            exec_ttft = None
            exec_tps = None
            if 'exec' in envs:
                pts = envs['exec']
                ttft_vals = [float(pt['ttft']) for pt in pts if pt['ttft'] != ""]
                tps_vals = [float(pt['tps']) for pt in pts if pt['tps'] != ""]
                if ttft_vals and tps_vals:
                    exec_ttft = statistics.mean(ttft_vals)
                    exec_tps = statistics.mean(tps_vals)
                    
            for env in ['exec', 'vm', 'linux']:
                if env in envs:
                    pts = envs[env]
                    ttft_vals = [float(pt['ttft']) for pt in pts if pt['ttft'] != ""]
                    tps_vals = [float(pt['tps']) for pt in pts if pt['tps'] != ""]
                    
                    if ttft_vals and tps_vals:
                        mean_ttft = statistics.mean(ttft_vals)
                        std_ttft = statistics.stdev(ttft_vals) if len(ttft_vals) > 1 else 0.0
                        mean_tps = statistics.mean(tps_vals)
                        std_tps = statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0
                        
                        diff_ttft = ""
                        diff_tps = ""
                        
                        if exec_ttft and env != 'exec':
                            diff = ((mean_ttft - exec_ttft) / exec_ttft) * 100
                            diff_ttft = f"{diff:+.2f}%"
                        elif env == 'exec':
                            diff_ttft = "Baseline"
                            
                        if exec_tps and env != 'exec':
                            diff = ((mean_tps - exec_tps) / exec_tps) * 100
                            diff_tps = f"{diff:+.2f}%"
                        elif env == 'exec':
                            diff_tps = "Baseline"
                            
                        writer.writerow([
                            model, env, 
                            f"{mean_ttft:.6f}", f"{std_ttft:.6f}", diff_ttft, 
                            f"{mean_tps:.4f}", f"{std_tps:.4f}", diff_tps
                        ])

    print("Created LaTeX Output files:")
    print(f"  - {perf_ind_file}")
    print(f"  - {perf_comp_file}")
    print(f"  - {eff_table_file}")
    print("Created Data files:")
    print(f"  - {means_report_file}")

if __name__ == "__main__":
    main()
