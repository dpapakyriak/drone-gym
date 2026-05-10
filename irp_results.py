import pandas as pd
import glob
import matplotlib.pyplot as plt

# Load all IRP CSV logs
files = glob.glob('./train_logs/loss_log_irp_*.csv')

dfs = []
for file in files:
    info = file.replace('./train_logs/loss_log_irp_', '').replace('.csv', '').split('_')
    num_nodes = int(info[0])
    seed = int(info[1])
    df = pd.read_csv(file)
    df['num_nodes'] = num_nodes
    df['seed'] = seed
    dfs.append(df)

data = pd.concat(dfs)
print(f"Loaded {len(files)} files")

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('IRP Training Curves (Cost over Epochs)', fontsize=14)

for ax, num_nodes in zip(axes, [20, 30, 40]):
    subset = data[data['num_nodes'] == num_nodes]
    avg = subset.groupby('Epoch')['Cost'].mean()

    for seed in subset['seed'].unique():
        seed_data = subset[subset['seed'] == seed]
        ax.plot(seed_data['Epoch'], seed_data['Cost'], alpha=0.3, color='blue', linewidth=0.8)

    ax.plot(avg.index, avg.values, color='blue', linewidth=2, label='Mean')
    ax.set_title(f'{num_nodes} Nodes')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost (negative distance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./train_logs/irp_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved to ./train_logs/irp_training_curves.png')

# Print final results summary
print('\n=== IRP Final Results (last 50 epochs average) ===')
for num_nodes in [20, 30, 40]:
    subset = data[data['num_nodes'] == num_nodes]
    final = subset[subset['Epoch'] >= 800].groupby('seed')['Cost'].mean()
    print(f'\n{num_nodes} nodes:')
    for seed, cost in final.items():
        print(f'  Seed {seed}: avg cost = {cost:.4f}')
    print(f'  Overall avg: {final.mean():.4f}')
