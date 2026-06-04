import pandas as pd
import os

def generate_benchmark_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "combined_general_stats_diff.csv")
    output_path = os.path.join(script_dir, "custom_benchmark_plot.tex")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for mean time
    df = df[df["Metric"] == "mean_s"].copy()
    
    # Shorten benchmark names for the X-axis
    name_map = {
        "CPU (Numpy)": "CPU (Numpy)",
        "CoreML (ALL)": "CoreML (All)",
        "CoreML (CPU_AND_GPU)": "CoreML (CPU+GPU)",
        "CoreML (CPU_ONLY)": "CoreML (CPU)",
        "GPU (MLX)": "GPU (MLX)",
        "PyTorch Tensor Math (MPS (Apple Silicon GPU))": "PyTorch (MPS)"
    }
    
    df["BenchmarkShort"] = df["Benchmark"].map(lambda x: name_map.get(x, x))
    
    benchmarks = df["BenchmarkShort"].tolist()
    
    exec_coords = []
    vm_coords = []
    
    for _, row in df.iterrows():
        b_name = row["BenchmarkShort"]
        # Convert seconds to ms
        exec_val = float(row["exec"]) * 1000 if pd.notna(row["exec"]) else 0
        vm_val = float(row["vm"]) * 1000 if pd.notna(row["vm"]) else 0
        
        exec_coords.append(f"({b_name}, {exec_val:.2f})")
        vm_coords.append(f"({b_name}, {vm_val:.2f})")
        
    exec_str = " ".join(exec_coords)
    vm_str = " ".join(vm_coords)
    
    b_names_str = ", ".join(benchmarks)
    
    latex_code = f"""\\begin{{figure}}[h!]
    \\centering
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
            ybar=1pt,
            bar width=12pt,
            width=0.9\\linewidth,
            height=7.5cm,
            enlarge x limits=0.15,
            ylabel={{Execution Time (ms)}},
            symbolic x coords={{{b_names_str}}},
            xtick=data,
            x tick label style={{rotate=25, anchor=east, font=\\small}},
            ymajorgrids=true,
            legend style={{
                at={{(0.5,-0.25)}},
                anchor=north,
                legend columns=-1,
                draw=none,
                /tikz/every even column/.style={{column sep=15pt}}
            }}
        ]
            \\addplot[fill=blue] coordinates {{{exec_str}}};
            \\addplot[fill=red] coordinates {{{vm_str}}};
            \\legend{{User-Space, Virtualization}}
        \\end{{axis}}
    \\end{{tikzpicture}}
    \\caption{{Performance comparison of custom benchmarks (Mean Execution Time), lower is better.}}
    \\label{{fig:custom-benchmark}}
\\end{{figure}}
"""

    with open(output_path, "w") as f:
        f.write(latex_code)
        
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_benchmark_plot()
