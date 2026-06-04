import pandas as pd
import numpy as np
import os

# Configuration
csv_path = "c:/Users/boink/Desktop/Repos/macos-ai-benchmarks/results/combined_mlx_benchmark_detailed.csv"
output_dir = "c:/Users/boink/Desktop/Repos/macos-ai-benchmarks/results"

def escape_latex(s):
    if pd.isna(s):
        return ""
    s = str(s)
    # Escape special LaTeX characters
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s

def generate_table(df, filename, include_diff=True, slice_size=None):
    # filter rows if not include_diff
    if not include_diff:
        df = df[df["Environment"] != "% Diff (VM vs Exec)"]
    
    # Process environment names
    df = df.copy()
    df["Environment"] = df["Environment"].replace({
        "exec": "User-space",
        "vm": "Virtualization",
        "% Diff (VM vs Exec)": "%Diff"
    })
    
    # Clean up column names
    cols = df.columns.tolist()
    latex_cols = [escape_latex(c) for c in cols]
    
    # We group by Operation
    operations = df["Operation"].dropna().unique()
    
    out_path = os.path.join(output_dir, filename)
    
    if slice_size is None:
        # Generate one longtable
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("% Requires \\usepackage{longtable} and \\usepackage{multirow}\n")
            f.write("\\begin{longtable}{|l|l|" + "r|" * (len(cols) - 2) + "}\n")
            f.write("\\caption{MLX Benchmark Results. Time is in ms, and \\%Diff is the time difference between Virtualization and User-space.}\\\\\n")
            f.write("\\hline\n")
            
            # Header
            header = " & ".join([f"\\textbf{{{c}}}" for c in latex_cols])
            f.write(f"{header} \\\\\n")
            f.write("\\hline\n")
            f.write("\\endfirsthead\n\n")
            
            f.write("\\multicolumn{" + str(len(cols)) + "}{c}\n")
            f.write("{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n")
            f.write("\\hline\n")
            f.write(f"{header} \\\\\n")
            f.write("\\hline\n")
            f.write("\\endhead\n\n")
            
            f.write("\\hline \\multicolumn{" + str(len(cols)) + "}{|r|}{{Continued on next page}} \\\\\n")
            f.write("\\hline\n")
            f.write("\\endfoot\n\n")
            
            f.write("\\hline\n")
            f.write("\\endlastfoot\n\n")
            
            for op in operations:
                op_df = df[df["Operation"] == op]
                num_rows = len(op_df)
                
                first_val = str(op_df.iloc[0]["Operation"])
                lines_in_op = 1 + first_val.count(" dim=") if " / " in first_val else 1
                span_rows = max(num_rows, lines_in_op)
                extra_rows = span_rows - num_rows

                for i, (_, row) in enumerate(op_df.iterrows()):
                    row_data = []
                    for j, col in enumerate(cols):
                        val = row[col]
                        if col == "Operation":
                            if i == 0:
                                val_esc = escape_latex(val)
                                if " / " in val_esc:
                                    parts = val_esc.split(" / ", 1)
                                    op_name = parts[0]
                                    dims = parts[1].replace(" dim=", " \\\\ dim=")
                                    op_formatted = f"\\begin{{tabular}}{{@{{}}l@{{}}}}{op_name} \\\\ {dims}\\end{{tabular}}"
                                else:
                                    op_formatted = val_esc
                                row_data.append(f"\\multirow{{{span_rows}}}{{*}}{{{op_formatted}}}")
                            else:
                                row_data.append("")
                        else:
                            val_str = str(val)
                            if "Inf" in val_str:
                                val_str = "-"
                            row_data.append(escape_latex(val_str))
                    f.write(" & ".join(row_data) + " \\\\\n")
                
                for _ in range(extra_rows):
                    f.write(" & " * (len(cols) - 1) + " \\\\\n")
                    
                f.write("\\hline\n")
            
            f.write("\\end{longtable}\n")
    else:
        # Sliced tables
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("% Requires \\usepackage{multirow} and \\usepackage{graphicx} (for resizebox)\n")
            num_ops = len(operations)
            
            chunks = []
            first_slice = 9  # Adjusted down to 9 just to be safe for the chapter heading
            if num_ops > first_slice:
                chunks.append(operations[:first_slice])
                for i in range(first_slice, num_ops, slice_size):
                    chunks.append(operations[i:i+slice_size])
            else:
                chunks.append(operations)
                
            for i, chunk_ops in enumerate(chunks):
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{MLX Benchmark Results (Part {i + 1}). Time is in ms, and \\%Diff is the time difference between Virtualization and User-space.}}\n")
                f.write("\\resizebox{\\textwidth}{!}{\n")
                f.write("\\begin{tabular}{|l|l|" + "r|" * (len(cols) - 2) + "}\n")
                f.write("\\hline\n")
                
                header = " & ".join([f"\\textbf{{{c}}}" for c in latex_cols])
                f.write(f"{header} \\\\\n")
                f.write("\\hline\n")
                
                for op in chunk_ops:
                    op_df = df[df["Operation"] == op]
                    num_rows = len(op_df)
                    
                    first_val = str(op_df.iloc[0]["Operation"])
                    lines_in_op = 1 + first_val.count(" dim=") if " / " in first_val else 1
                    span_rows = max(num_rows, lines_in_op)
                    extra_rows = span_rows - num_rows

                    for k, (_, row) in enumerate(op_df.iterrows()):
                        row_data = []
                        for j, col in enumerate(cols):
                            val = row[col]
                            if col == "Operation":
                                if k == 0:
                                    val_esc = escape_latex(val)
                                    if " / " in val_esc:
                                        parts = val_esc.split(" / ", 1)
                                        op_name = parts[0]
                                        dims = parts[1].replace(" dim=", " \\\\ dim=")
                                        op_formatted = f"\\begin{{tabular}}{{@{{}}l@{{}}}}{op_name} \\\\ {dims}\\end{{tabular}}"
                                    else:
                                        op_formatted = val_esc
                                    row_data.append(f"\\multirow{{{span_rows}}}{{*}}{{{op_formatted}}}")
                                else:
                                    row_data.append("")
                            else:
                                val_str = str(val)
                                if "Inf" in val_str:
                                    val_str = "-"
                                row_data.append(escape_latex(val_str))
                        f.write(" & ".join(row_data) + " \\\\\n")
                    
                    for _ in range(extra_rows):
                        f.write(" & " * (len(cols) - 1) + " \\\\\n")
                        
                    f.write("\\hline\n")
                
                f.write("\\end{tabular}\n")
                f.write("}\n")
                f.write("\\end{table}\n\n")

if __name__ == '__main__':
    df = pd.read_csv(csv_path)
    
    # Remove empty rows
    df = df.dropna(subset=["Environment"])
    
    # Drop speedup columns to reduce table width
    speedup_cols = [c for c in df.columns if 'speedup' in c.lower()]
    df = df.drop(columns=speedup_cols)
    
    # Reorder columns: Operation, Environment, ...
    cols = df.columns.tolist()
    if "Operation" in cols:
        cols.remove("Operation")
        cols.insert(0, "Operation")
        df = df[cols]
    
    print("Generating table with diff (longtable)...")
    generate_table(df, "mlx_benchmark_with_diff.tex", include_diff=True)
    
    print("Generating table without diff (longtable)...")
    generate_table(df, "mlx_benchmark_without_diff.tex", include_diff=False)
    
    print("Generating sliced tables (regular table)...")
    generate_table(df, "mlx_benchmark_sliced.tex", include_diff=True, slice_size=13)
    
    print("Done! Files saved in", output_dir)
