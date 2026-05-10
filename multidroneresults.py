import pandas as pd
import glob
import matplotlib.pyplot as plt
import re

files = glob.glob('./train_logs/*loss_log_multi_*drones_*.csv')
print(f"Found files: {files}")

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
print(f"Loaded {len(files)} training logs\n")

print('=== Multi-Drone Final Results (last 50 epochs avg) ===')
for k in sorted(data['k'].unique()):
    print(f'\n--- {k} Drones ---')
    for max_range in sorted(data['max_range'].unique()):
        print(f'  Range = {max_range}:')
        subset = data[(data['k'] == k) & (data['max_range'] == max_range)]
        for num_nodes in sorted(subset['num_nodes'].unique()):
            node_subset = subset[subset['num_nodes'] == num_nodes]
            final = node_subset[node_subset['Epoch'] >= 800].groupby('seed')['Cost'].mean()
            print(f'    {num_nodes} nodes: avg cost = {final.mean():.4f}')