import os
import csv
import glob
from collections import defaultdict

def parse_txt_to_lists(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
    detailed_lines = []
    average_lines = []
    current_list = None
    
    for line in lines:
        line = line.strip()
        if "Detailed benchmark" in line:
            current_list = detailed_lines
            continue
        elif "Average benchmark" in line:
            current_list = average_lines
            continue
            
        if line.startswith("|") and current_list is not None:
            # Skip markdown separator lines
            if "---" in line:
                continue
            
            # Split and strip row values
            row = [item.strip() for item in line.split("|")[1:-1]]
            current_list.append(row)
            
    return detailed_lines, average_lines

def create_diff_master(master_rows, output_path):
    if not master_rows: 
        return
    
    header = master_rows[0]
    data = master_rows[1:]
    
    # Group by Operation (index 1)
    by_op = defaultdict(list)
    for row in data:
        by_op[row[1]].append(row)
        
    final_data = []
    for op in sorted(by_op.keys()):
        rows = by_op[op]
        # Sort so 'exec' appears before 'vm'
        rows.sort(key=lambda x: x[0]) 
        final_data.extend(rows)
        
        # Calculate % difference if both exec and vm exist
        exec_row = next((r for r in rows if r[0] == "exec"), None)
        vm_row = next((r for r in rows if r[0] == "vm"), None)
        
        if exec_row and vm_row:
            # Create a difference row
            diff_row = ["% Diff (VM vs Exec)", op]
            for i in range(2, len(exec_row)):
                try:
                    val_exec = float(exec_row[i])
                    val_vm = float(vm_row[i])
                    
                    if val_exec == 0:
                        diff_row.append("Inf%")
                    else:
                        diff = ((val_vm - val_exec) / val_exec) * 100.0
                        # Positive diff = VM took more time (slower)
                        # Negative diff = VM took less time (faster) 
                        sign = "+" if diff > 0 else ""
                        diff_row.append(f"{sign}{diff:.2f}%")
                except ValueError:
                    # Ignore columns that contain text/nan/already have % signs
                    diff_row.append("-")
            final_data.append(diff_row)
            # Add an empty row for visual spacing between operation groups
            final_data.append([""] * len(header))
            
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_data)
    print(f"Created master file: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_pattern = os.path.join(script_dir, "**", "mlx_benchmark_suite_mac.txt")
    txt_files = glob.glob(search_pattern, recursive=True)
    
    if not txt_files:
        print("No mlx_benchmark_suite_mac.txt files found in subdirectories.")
        exit(0)
        
    master_detailed_rows = []
    master_average_rows = []
    
    for file_path in txt_files:
        env_name = os.path.basename(os.path.dirname(file_path))
        
        det_lines, avg_lines = parse_txt_to_lists(file_path)
        base_dir = os.path.dirname(file_path)
        
        indiv_det_csv = os.path.join(base_dir, f"{env_name}_mlx_benchmark_detailed.csv")
        indiv_avg_csv = os.path.join(base_dir, f"{env_name}_mlx_benchmark_average.csv")
        
        if det_lines:
            with open(indiv_det_csv, 'w', newline='') as f:
                csv.writer(f).writerows(det_lines)
            print(f"Created {indiv_det_csv}")
            
            header = ["Environment"] + det_lines[0]
            if not master_detailed_rows:
                master_detailed_rows.append(header)
            for row in det_lines[1:]:
                master_detailed_rows.append([env_name] + row)
                
        if avg_lines:
            with open(indiv_avg_csv, 'w', newline='') as f:
                csv.writer(f).writerows(avg_lines)
            print(f"Created {indiv_avg_csv}")
            
            header = ["Environment"] + avg_lines[0]
            if not master_average_rows:
                master_average_rows.append(header)
            for row in avg_lines[1:]:
                master_average_rows.append([env_name] + row)

    # Process and write the master files with % difference rows
    master_det_csv = os.path.join(script_dir, "combined_mlx_benchmark_detailed.csv")
    create_diff_master(master_detailed_rows, master_det_csv)

    master_avg_csv = os.path.join(script_dir, "combined_mlx_benchmark_average.csv")
    create_diff_master(master_average_rows, master_avg_csv)
