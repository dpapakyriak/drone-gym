from gym_vrp.envs import MultiDroneVRPEnv
from agents import MultiDroneVRPAgent
import os

max_range = 5.0
num_drones = 10  # change to 5 or 10 for other runs
seeds = [69, 123]
num_nodes = [20, 30, 40]
batch_size = 128

os.makedirs('./train_logs', exist_ok=True)

for seed in seeds:
    for num_node in num_nodes:
        print(f'Training: {num_node} nodes, seed {seed}, {num_drones} drones, max_range {max_range}')
        env = MultiDroneVRPEnv(
            num_nodes=num_node,
            batch_size=batch_size,
            seed=seed,
            max_range=max_range,
            num_drones=num_drones
        )
        MultiDroneVRPAgent(
            seed=seed,
            csv_path=f'./train_logs/loss_log_multi_{num_drones}drones_{num_node}_{seed}.csv'
        ).train(
            env,
            epochs=851,
            check_point_dir=f'./check_points/multi_{num_drones}drones_{num_node}_{seed}/'
        )
