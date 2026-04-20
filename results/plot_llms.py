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
    models = df['Model'].unique().tolist()
    
    # Map each model to a distinct shape (optional now that they are separate, but good for consistency)
    markers = ['o', '^', 'D', 's', 'v', 'P', '*']
    model_marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}
    
    # Environments get different colors
    color_map = {'exec': '#2ca02c', 'vm': '#d62728'} # Green and Red
    
    for model in models:
        # Create a fresh figure for this specific model
        plt.figure(figsize=(10, 7))
        
        for env in ['exec', 'vm']:
            subset = df[(df['Model'] == model) & (df['Environment'] == env)]
            if subset.empty: 
                continue
            
            plt.scatter(
                subset['Time_To_First_Token_S'],
                subset['Tokens_Per_Second'],
                c=color_map[env],
                marker=model_marker_map[model],
                s=150,               # marker size
                alpha=0.75,          # transparency
                edgecolors='black',  # black border
                linewidths=1.5,
                label=f"{env.upper()}"
            )
            
        plt.title(f"LLM Performance: {model}", fontsize=16, fontweight='bold')
        plt.xlabel("Time to First Token (Seconds)  ← LOWER IS BETTER", fontsize=13)
        plt.ylabel("Tokens Per Second  ↑ HIGHER IS BETTER", fontsize=13)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Legend formatting
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Sanitize model name for the filename
        safe_model_name = model.replace("/", "_").replace("\\", "_")
        out_img = os.path.join(script_dir, f"llm_scatter_{safe_model_name}.png")
        
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to prevent them from stacking up and saving over each other!
        
        print(f"Successfully generated graph: {out_img}")

if __name__ == "__main__":
    main()
