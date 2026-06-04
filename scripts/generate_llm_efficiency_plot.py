import os
import json
import glob
import statistics
from collections import defaultdict

def generate_llm_efficiency_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "llm_efficiency_plot.tex")
    
    search_pattern = os.path.join(script_dir, "**", "llm_stats_*.json")
    json_files = glob.glob(search_pattern, recursive=True)
    
    # data[model][env] = [tpw1, tpw2, ...]
    data = defaultdict(lambda: defaultdict(list))
    
    for file_path in json_files:
        if "old" in file_path.lower().split(os.sep):
            continue
            
        filename = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        env_name = None
        if "exec" in dir_name.lower():
            env_name = "User-space"
        elif "vm" in dir_name.lower():
            env_name = "Virtualization"
        elif "linux" in dir_name.lower():
            env_name = "H100"
            
        if not env_name:
            continue
            
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if isinstance(content, dict) and "runs" in content:
                    model_name = content.get("model", filename)
                    model_name = model_name.split("/")[-1]
                    model_name = model_name.replace("-Instruct", "").replace("-4bit", "").replace("_OPTIMIZED.json", "").replace("_NATIVE.json", "").replace(".json", "")
                    if model_name.startswith("llm_stats_"):
                        model_name = model_name.replace("llm_stats_", "")
                    
                    runs = content["runs"]
                    for run in runs:
                        tpw = run.get("tokens_per_watt", None)
                        if tpw is not None:
                            data[model_name][env_name].append(float(tpw))
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")

    models = sorted(data.keys())
    if not models:
        print("No valid TPW data found.")
        return
        
    # Compute means
    means = defaultdict(dict)
    for model in models:
        for env in ["User-space", "Virtualization", "H100"]:
            if data[model][env]:
                means[env][model] = statistics.mean(data[model][env])
            else:
                means[env][model] = 0.0

    def get_coords(env):
        coords = []
        for model in models:
            val = means[env][model]
            coords.append(f"({model}, {val:.4f})")
        return " ".join(coords)
        
    exec_coords = get_coords("User-space")
    vm_coords = get_coords("Virtualization")
    h100_coords = get_coords("H100")
    
    models_str = ", ".join(models)
    
    latex_code = f"""\\begin{{figure}}[h!]
    \\centering
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
            ybar=1pt,
            bar width=15pt,
            width=0.9\\linewidth,
            height=7.5cm,
            enlarge x limits=0.2,
            ylabel={{Average TPW (Tokens Per Watt)}},
            symbolic x coords={{{models_str}}},
            xtick=data,
            x tick label style={{font=\\small}},
            ymajorgrids=true,
            legend style={{
                at={{(0.5,-0.15)}},
                anchor=north,
                legend columns=-1,
                draw=none,
                /tikz/every even column/.style={{column sep=15pt}}
            }}
        ]
            \\addplot[fill=blue] coordinates {{{exec_coords}}};
            \\addplot[fill=red] coordinates {{{vm_coords}}};
            \\addplot[fill=green] coordinates {{{h100_coords}}};
            \\legend{{User-space, Virtualization, H100}}
        \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{LLM Efficiency Comparison (Average Tokens Per Watt), higher is better.}}
    \\label{{fig:llm_efficiency}}
\\end{{figure}}
"""

    with open(output_path, "w") as f:
        f.write(latex_code)
        
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_llm_efficiency_plot()
