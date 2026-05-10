from gym_vrp.envs import IRPEnv
from agents import IRPAgent

seeds = [69, 123]
num_nodes = [20, 30, 40]
batch_size = 256

for seed in seeds:
    for num_node in num_nodes:
        env = IRPEnv(num_nodes=num_node, batch_size=batch_size, seed=seed)
        IRPAgent(seed=seed, csv_path=f'./train_logs/loss_log_irp_{num_node}_{seed}.csv').train(
            env, epochs=851, check_point_dir=f'./check_points/irp_{num_node}_{seed}/'
        )