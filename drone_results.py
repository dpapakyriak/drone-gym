import pandas as pd
import glob
import matplotlib.pyplot as plt

max_range = 5.0  # change to match which range you want to visualize

files = glob.glob(f'./train_logs/loss_log_drone_*_range{max_range}.csv')
print(f"Found files: {files}")

dfs = []
for file in files:
    parts = file.replace(f'./train_logs/loss_log_drone_', '').replace(f'_range{max_range}.csv', '').split('_')
    num_nodes = int(parts[0])
    seed = int(parts[1])
    df = pd.read_csv(file)
    df['num_nodes'] = num_nodes
    df['seed'] = seed
    dfs.append(df)

data = pd.concat(dfs)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'Drone VRP Training Curves (max_range={max_range})', fontsize=14)

for ax, num_nodes in zip(axes, [20, 30, 40]):
    subset = data[data['num_nodes'] == num_nodes]
    avg = subset.groupby('Epoch')['Cost'].mean()
    for seed in subset['seed'].unique():
        seed_data = subset[subset['seed'] == seed]
        ax.plot(seed_data['Epoch'], seed_data['Cost'], alpha=0.4, linewidth=0.8, label=f'Seed {seed}')
    ax.plot(avg.index, avg.values, color='black', linewidth=2, linestyle='--', label='Mean')
    ax.set_title(f'{num_nodes} Nodes')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost (negative distance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'./train_logs/drone_training_curves_range{max_range}.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
print(f'\n=== Drone VRP Final Results (max_range={max_range}, last 50 epochs avg) ===')
for num_nodes in [20, 30, 40]:
    subset = data[data['num_nodes'] == num_nodes]
    final = subset[subset['Epoch'] >= 800].groupby('seed')['Cost'].mean()
    print(f'\n{num_nodes} nodes:')
    for seed, cost in final.items():
        print(f'  Seed {seed}: avg cost = {cost:.4f}')
    print(f'  Overall avg: {final.mean():.4f}')