import os
import glob
import re
import csv
import argparse
from collections import defaultdict

def parse_fio_file(filepath):
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by test sections demarcated by "--- X. Test Name ---"
    sections = re.split(r'---\s*\d+\.\s*(.*?)\s*---', content)
    
    for i in range(1, len(sections), 2):
        test_name = sections[i].strip()
        test_output = sections[i+1]
        
        # 1. Parse IOPS
        iops_matches = re.findall(r'(?:read|write):\s*IOPS=([\d\.]+[kKmM]?)', test_output)
        
        total_iops = 0
        for iops_str in iops_matches:
            if iops_str.lower().endswith('k'):
                total_iops += int(float(iops_str[:-1]) * 1000)
            elif iops_str.lower().endswith('m'):
                total_iops += int(float(iops_str[:-1]) * 1000000)
            else:
                total_iops += int(float(iops_str))
                
        # 2. Parse Aggregate Bandwidth
        agg_match = re.search(r'Run status group 0 \(all jobs\):\s*\n\s*(?:READ|WRITE):\s*bw=([\d\.]+[a-zA-Z]+/s)', test_output)
        if not agg_match:
            # Fallback if there's only 1 job or slightly different format
            agg_match = re.search(r'(?:read|write):\s*IOPS=[\d\.]+[kKmM]?,\s*BW=([\d\.]+[a-zA-Z]+/s)', test_output)
        
        agg_bw = agg_match.group(1) if agg_match else "N/A"
        
        # 3. Parse Average Latency
        # Matches: lat (usec): min=5, max=1347.3k, avg=7081.20, stdev=26967.71
        lat_matches = re.findall(r'\s+lat \((msec|usec|nsec)\):\s*min=[^,]+,\s*max=[^,]+,\s*avg=([\d\.]+[kKmM]?)', test_output)
        avg_lat_ms = "N/A"
        if lat_matches:
            total_lat_ms = 0
            for unit, avg in lat_matches:
                avg_val_str = avg
                multiplier = 1.0
                if avg_val_str.lower().endswith('k'):
                    avg_val_str = avg_val_str[:-1]
                    multiplier = 1000.0
                elif avg_val_str.lower().endswith('m'):
                    avg_val_str = avg_val_str[:-1]
                    multiplier = 1000000.0
                    
                avg_val = float(avg_val_str) * multiplier
                
                if unit == 'usec':
                    avg_val /= 1000.0
                elif unit == 'nsec':
                    avg_val /= 1000000.0
                    
                total_lat_ms += avg_val
                
            avg_lat_ms = round(total_lat_ms / len(lat_matches), 2)

        results[test_name] = {
            'IOPS': total_iops,
            'BW': agg_bw,
            'AvgLat_ms': avg_lat_ms
        }
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='', help='Optional date string to filter by (e.g., 2026-04-20)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_pattern = f"fio_*result*{args.date}*.txt" if args.date else "fio_*result*.txt"
    search_pattern = os.path.join(script_dir, "**", file_pattern)
    txt_files = glob.glob(search_pattern, recursive=True)

    if not txt_files:
        print("No fio benchmark result files found.")
        return

    master_data = []
    
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        if 'fio_vm_' in filename:
            env = 'vm'
        elif 'native_ceph' in filename:
            env = 'native_ceph'
        else:
            env = 'exec'
            
        # Try to extract a date/timestamp if available
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})', filename)
        date_str = date_match.group(1) if date_match else "Unknown"
            
        parsed_data = parse_fio_file(file_path)
        for test_name, metrics in parsed_data.items():
            master_data.append([
                env, date_str, test_name, metrics['IOPS'], metrics['BW'], metrics['AvgLat_ms']
            ])
            
    # Sort to neatly align environments
    master_data.sort(key=lambda x: (x[2], x[0], x[1]))
            
    # Write CSV
    out_csv = os.path.join(script_dir, "combined_fio_data.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Environment", "Date", "Test", "IOPS", "Bandwidth", "AvgLatency_ms"])
        writer.writerows(master_data)
        
    print(f"Created FIO data file: {out_csv}")
    
    # Write LaTeX Table
    # Group by Test, then Environment
    tests = {}
    for row in master_data:
        env = row[0]
        test = row[2]
        if test not in tests:
            tests[test] = {}
        # Taking the latest or simply overwriting if multiple dates exist
        tests[test][env] = {
            'IOPS': row[3],
            'BW': row[4],
            'AvgLat_ms': row[5]
        }
        
    latex_file = os.path.join(script_dir, "fio_latex_table.tex")
    with open(latex_file, 'w') as f:
        f.write("% Add to your preamble: \\usepackage{multirow}\n")
        f.write("\\begin{table}[hbt!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|p{4cm}|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Test} & \\textbf{Environment} & \\textbf{IOPS} & \\textbf{Bandwidth} & \\textbf{Avg Latency (ms)} \\\\\n")
        f.write("\\hline\n")
        
        for test, env_data in tests.items():
            envs = sorted(list(env_data.keys()))
            num_envs = len(envs)
            
            # Make sure special characters in Test names are escaped
            safe_test = test.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
            
            for i, env in enumerate(envs):
                if env == "vm":
                    env_label = "Virtualization"
                elif env == "native_ceph":
                    env_label = "Native Ceph"
                else:
                    env_label = "User-space"
                    
                iops = env_data[env]['IOPS']
                bw = env_data[env]['BW']
                lat = env_data[env]['AvgLat_ms']
                
                if i == 0:
                    test_col = f"\\multirow{{{num_envs}}}{{*}}{{\\parbox{{4cm}}{{{safe_test}}}}}"
                else:
                    test_col = ""
                
                f.write(f"{test_col} & {env_label} & {iops} & {bw} & {lat} \\\\\n")
                
            f.write("\\hline\n")
            
        f.write("\\end{tabular}\n")
        f.write("\\caption{FIO Storage Benchmark Results Comparison}\n")
        f.write("\\label{tab:fio_results}\n")
        f.write("\\end{table}\n")
        
    print(f"Created LaTeX FIO Table: {latex_file}")

if __name__ == "__main__":
    main()
