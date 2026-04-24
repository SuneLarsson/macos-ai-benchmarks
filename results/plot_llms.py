import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "combined_llm_scatter_data.csv")
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # Clean up any potential whitespace
    df['Environment'] = df['Environment'].str.strip()
    df['Model'] = df['Model'].str.strip()
    
    models = df['Model'].unique().tolist()
    
    # Map each model to a distinct shape (4 models = 4 shapes)
    markers = ['o', '^', 's', 'D', 'v', 'P', '*']
    model_marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}
    
    # Environments get different colors (blue for exec, red for vm)
    color_map = {'exec': 'blue', 'vm': 'red', 'linux': 'green'}
    
    plt.figure(figsize=(14, 8))
    
    for model in models:
        for env in ['exec', 'vm', 'linux']:
            subset = df[(df['Model'] == model) & (df['Environment'] == env)]
            if subset.empty: 
                continue
            
            plt.scatter(
                subset['Time_To_First_Token_S'],
                subset['Tokens_Per_Second'],
                c=color_map.get(env, 'gray'),
                marker=model_marker_map[model],
                s=150,               # marker size
                alpha=0.75,          # transparency
                edgecolors='black',  # black border
                linewidths=1.5,
                label=f"{model} ({env.upper()})"
            )
            
    plt.title("LLM Performance Comparison (All Models)", fontsize=18, fontweight='bold')
    plt.xlabel("Time to First Token (Seconds)  ← LOWER IS BETTER", fontsize=14)
    plt.ylabel("Tokens Per Second  ↑ HIGHER IS BETTER", fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Start both axes at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Legend formatting
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    
    out_img = os.path.join(script_dir, "llm_scatter_all_models.png")
    
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"Successfully generated combined graph: {out_img}")

if __name__ == "__main__":
    main()
