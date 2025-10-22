import re
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def parse_training_log(log_file):
    """Parse the training log to extract metrics."""
    
    metrics_data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for lines containing training metrics
            if 'training/global_step:' in line:
                metrics = {}
                
                # Extract step number
                step_match = re.search(r'training/global_step:(\d+)', line)
                if step_match:
                    metrics['step'] = int(step_match.group(1))
                else:
                    continue
                
                # Extract critic score mean
                critic_score = re.search(r'critic/score/mean:([\d.]+)', line)
                if critic_score:
                    metrics['critic_score_mean'] = float(critic_score.group(1))
                
                # Extract response length mean
                resp_mean = re.search(r'response_length/mean:([\d.]+)', line)
                if resp_mean:
                    metrics['response_length_mean'] = float(resp_mean.group(1))
                
                # Extract response length clip ratio
                resp_clip_ratio = re.search(r'response_length/clip_ratio:([\d.]+)', line)
                if resp_clip_ratio:
                    metrics['response_length_clip_ratio'] = float(resp_clip_ratio.group(1))
                
                if metrics:  # Only add if we found at least the step
                    metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)

def create_plots(log_file, output_dir):
    """Create the 3 requested plots from training metrics."""
    
    # Parse the log file
    df = parse_training_log(log_file)
    
    if df.empty:
        print("No metrics found in log file!")
        print("Please check if the log file contains lines with 'training/global_step:'")
        return
    
    print(f"Found {len(df)} training steps with metrics")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Metrics Over Steps', fontsize=16, y=1.05)
    
    # Plot 1: Critic Score Mean
    ax1 = axes[0]
    if 'critic_score_mean' in df.columns and df['critic_score_mean'].notna().any():
        ax1.plot(df['step'], df['critic_score_mean'], 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Score')
        ax1.set_title('Critic Score Mean over Training Steps')
        ax1.grid(True, alpha=0.3)
        print(f"✓ Plotted critic score mean ({df['critic_score_mean'].notna().sum()} points)")
    else:
        ax1.text(0.5, 0.5, 'No critic score data found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Critic Score Mean over Training Steps (No Data)')
    
    # Plot 2: Response Length Clip Ratio
    ax2 = axes[1]
    if 'response_length_clip_ratio' in df.columns and df['response_length_clip_ratio'].notna().any():
        ax2.plot(df['step'], df['response_length_clip_ratio'], 'g-', linewidth=2, 
                marker='^', markersize=4)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Clip Ratio')
        ax2.set_title('Response Length Clip Ratio over Training Steps')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])  # Clip ratio should be between 0 and 1
        print(f"✓ Plotted response length clip ratio ({df['response_length_clip_ratio'].notna().sum()} points)")
    else:
        ax2.text(0.5, 0.5, 'No response length clip ratio data found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Response Length Clip Ratio over Training Steps (No Data)')
    
    # Plot 3: Response Length Mean
    ax3 = axes[2]
    if 'response_length_mean' in df.columns and df['response_length_mean'].notna().any():
        ax3.plot(df['step'], df['response_length_mean'], 'purple', linewidth=2, 
                marker='D', markersize=4)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Length (tokens)')
        ax3.set_title('Response Length Mean over Training Steps')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=256, color='r', linestyle='--', alpha=0.5, label='Max Length (256)')
        ax3.legend()
        print(f"✓ Plotted response length mean ({df['response_length_mean'].notna().sum()} points)")
    else:
        ax3.text(0.5, 0.5, 'No response length data found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Response Length Mean over Training Steps (No Data)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plots saved to {plot_path}")
    
    # Also save the data as CSV
    csv_path = output_path / 'metrics_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Metrics data saved to {csv_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print("-" * 40)
    if 'critic_score_mean' in df.columns and df['critic_score_mean'].notna().any():
        print(f"Steps analyzed: {len(df)}")
        print(f"Final Critic Score: {df['critic_score_mean'].iloc[-1]:.6f}")
        print(f"Mean Critic Score: {df['critic_score_mean'].mean():.6f}")
        print(f"Max Critic Score: {df['critic_score_mean'].max():.6f}")
    
    if 'response_length_mean' in df.columns and df['response_length_mean'].notna().any():
        print(f"Final Avg Response Length: {df['response_length_mean'].iloc[-1]:.1f} tokens")
        print(f"Mean Response Length: {df['response_length_mean'].mean():.1f} tokens")
        print(f"Max Response Length: {df['response_length_mean'].max():.1f} tokens")
        print(f"Min Response Length: {df['response_length_mean'].min():.1f} tokens")
    
    if 'response_length_clip_ratio' in df.columns and df['response_length_clip_ratio'].notna().any():
        print(f"Final Response Length Clip Ratio: {df['response_length_clip_ratio'].iloc[-1]:.4f}")
        print(f"Mean Response Length Clip Ratio: {df['response_length_clip_ratio'].mean():.4f}")
    
    # Print what columns were found
    print(f"\nColumns found in data: {list(df.columns)}")
    print(f"Data shape: {df.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_metrics.py <log_file_path> <output_directory>")
        print("Example: python plot_metrics.py /workspace/results/qwen0.5b_temp1.0_20251015_084133/training.log /workspace/results/qwen0.5b_temp1.0_20251015_084133")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found!")
        sys.exit(1)
    
    print(f"Parsing log file: {log_file}")
    create_plots(log_file, output_dir)