import logging
from typing import Tuple

import torch

from .graph_encoder import GraphDemandEncoder
from .graph_tsp_agent import TSPAgent, TSPModel
from .drone_agent import DroneVRPModel

logging.basicConfig(level=logging.INFO)


class MultiDroneVRPModel(DroneVRPModel):
    """
    MultiDroneVRPModel extends DroneVRPModel to handle multiple drones.
    At each timestep, the decoder is called sequentially for each drone.
    Each drone gets its own context (load, battery) and its own mask.
    The encoder runs once per episode — the graph structure doesn't change.
    Drones operate independently (no inter-drone awareness).
    """

    def forward(self, env, rollout=False) -> Tuple[float, float]:
        """
        Forward method for multi-drone routing.

        Args:
            env: MultiDroneVRPEnv instance.
            rollout (bool): If True, act greedily. Defaults to False.

        Returns:
            Tuple[float, float]: accumulated loss and log probabilities.
        """
        done = False

        graph_state, load, battery, drone_masks = env.get_state()
        graph_state = torch.tensor(graph_state, dtype=torch.float, device=self.device)
        load = torch.tensor(load, dtype=torch.float, device=self.device)
        battery = torch.tensor(battery, dtype=torch.float, device=self.device)
        drone_masks = torch.tensor(drone_masks, dtype=torch.float, device=self.device)

        batch_size = graph_state.shape[0]
        num_drones = load.shape[1]

        acc_loss = torch.zeros(size=(batch_size,), device=self.device)
        acc_log_prob = torch.zeros(size=(batch_size,), device=self.device)

        # encode graph once — structure doesn't change during episode
        emb = self.encoder(
            x=graph_state[:, :, :3],
            depot_mask=graph_state[:, :, 3].bool()
        )

        while not done:
            # collect actions for all drones this timestep
            all_actions = torch.zeros(
                (batch_size, num_drones), dtype=torch.long, device=self.device
            )

            for d in range(num_drones):
                # get this drone's context
                drone_load = load[:, d]      # (batch_size,)
                drone_battery = battery[:, d]  # (batch_size,)
                drone_mask = drone_masks[:, d, :]  # (batch_size, num_nodes)

                # combine load and battery into single context scalar
                combined = torch.stack([drone_load, drone_battery], dim=1)
                vehicle_state = self.battery_proj(combined).squeeze(-1)

                actions, log_prob = self.decoder(
                    node_embs=emb,
                    mask=drone_mask,
                    load=vehicle_state,
                    rollout=rollout,
                )

                all_actions[:, d] = actions.squeeze()
                acc_log_prob += log_prob.squeeze().to(self.device)

            # step environment with all drone actions
            state, loss, done, _ = env.step(all_actions.cpu().numpy())

            acc_loss += torch.tensor(loss, dtype=torch.float, device=self.device)

            # update state
            graph_state, load, battery, drone_masks = state
            graph_state = torch.tensor(graph_state, dtype=torch.float, device=self.device)
            load = torch.tensor(load, dtype=torch.float, device=self.device)
            battery = torch.tensor(battery, dtype=torch.float, device=self.device)
            drone_masks = torch.tensor(drone_masks, dtype=torch.float, device=self.device)

        self.decoder.reset()

        return acc_loss, acc_log_prob


class MultiDroneVRPAgent(TSPAgent):
    """
    MultiDroneVRPAgent solves the multi-drone VRP with payload
    and battery constraints. Uses sequential decision making —
    each drone decides independently at each timestep.
    """

    def __init__(
        self,
        depot_dim: int = 2,
        node_dim: int = 3,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        lr: float = 1e-4,
        csv_path: str = "loss_log.csv",
        seed: int = 69,
    ):
        super().__init__(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            lr=lr,
            csv_path=csv_path,
            seed=seed,
        )

        self.model = MultiDroneVRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model = MultiDroneVRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
