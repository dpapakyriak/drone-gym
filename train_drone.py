from gym_vrp.envs import DroneVRPEnv
from agents import DroneVRPAgent
import os

max_range = 5.0  # change to 2.0 or 5.0 for other runs
seeds = [69, 123]
num_nodes = [20, 30, 40]
batch_size = 256

for seed in seeds:
    for num_node in num_nodes:
        print(f'Training: {num_node} nodes, seed {seed}, max_range {max_range}')
        env = DroneVRPEnv(num_nodes=num_node, batch_size=batch_size, seed=seed, max_range=max_range)
        DroneVRPAgent(
            seed=seed,
            csv_path=f'./train_logs/loss_log_drone_{num_node}_{seed}_range{max_range}.csv'
        ).train(
            env,
            epochs=851,
            check_point_dir=f'./check_points/drone_{num_node}_{seed}_range{max_range}/'
        )