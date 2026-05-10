import pandas as pd
import glob
import matplotlib.pyplot as plt
import re

files = glob.glob('./train_logs/*loss_log_multi_*drones_*.csv')

dfs = []
for file in files:
    basename = file.replace('./train_logs/', '')
    match = re.match(r'(\d+\.?\d*)loss_log_multi_(\d+)drones_(\d+)_(\d+)\.csv', basename)
    if match:
        max_range = float(match.group(1))
        k = int(match.group(2))
        num_nodes = int(match.group(3))
        seed = int(match.group(4))
        df = pd.read_csv(file)
        df['max_range'] = max_range
        df['k'] = k
        df['num_nodes'] = num_nodes
        df['seed'] = seed
        dfs.append(df)

data = pd.concat(dfs)

for k in sorted(data['k'].unique()):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Multi-Drone Training Curves ({k} Drones)', fontsize=14)
    
    for ax, max_range in zip(axes, sorted(data['max_range'].unique())):
        subset = data[(data['k'] == k) & (data['max_range'] == max_range)]
        for num_nodes in sorted(subset['num_nodes'].unique()):
            node_data = subset[subset['num_nodes'] == num_nodes]
            avg = node_data.groupby('Epoch')['Cost'].mean()
            for seed in node_data['seed'].unique():
                seed_data = node_data[node_data['seed'] == seed]
                ax.plot(seed_data['Epoch'], seed_data['Cost'], 
                       alpha=0.4, linewidth=0.8)
            ax.plot(avg.index, avg.values, linewidth=2, 
                   linestyle='--', color='black', label=f'{num_nodes} nodes mean')
        ax.set_title(f'R = {max_range}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost (negative distance)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'./train_logs/multidrone_{k}drones_curves.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')