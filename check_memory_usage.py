import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",                 
    "font.size": 12,
    "axes.labelsize": 14,                 
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2
})

def plot_hardware_usage(csv_file="hardware_usage.csv", output_img="hardware_graph.png"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Please run your main pipeline first!")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#1f77b4'
    ax1.set_xlabel('Execution Time (Seconds)', weight='bold', labelpad=10)
    ax1.set_ylabel('CPU Utilization (%)', color=color1, weight='bold', labelpad=10)
    
    line1 = ax1.plot(df['Time_Seconds'], df['CPU_Percent'], color=color1, linewidth=2, label='CPU (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax1.set_ylim(0, 105) 
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())

    ax2 = ax1.twinx()  
    color2 = '#d62728' 
    ax2.set_ylabel('RAM Usage (GB)', color=color2, weight='bold', labelpad=10)
    
    line2 = ax2.plot(df['Time_Seconds'], df['RAM_GB'], color=color2, linewidth=2.5, linestyle='--', label='RAM (GB)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    max_ram = df['RAM_GB'].max()
    padding = max_ram * 0.2 if max_ram > 0 else 1
    ax2.set_ylim(0, max_ram + padding)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.95, edgecolor='black', shadow=True)

    plt.title('System Resource Utilization During Pipeline Execution', pad=20, weight='bold', fontsize=16)

    fig.tight_layout()

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! High-resolution academic graph saved as: {output_img}")

if __name__ == "__main__":
    plot_hardware_usage()