import logging
from typing import Tuple

import torch

from .graph_encoder import GraphDemandEncoder
from .graph_tsp_agent import TSPAgent, TSPModel

logging.basicConfig(level=logging.INFO)


class DroneVRPModel(TSPModel):
    def __init__(
        self,
        depot_dim: int,
        node_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        num_heads: int,
    ):
        """
        DroneVRPModel extends TSPModel to solve the drone VRP with
        both a payload constraint (load) and a battery/range constraint.

        Args:
            depot_dim (int): Input dimension of a depot node.
            node_dim (int): Input dimension of a regular graph node.
            emb_dim (int): Size of a vector in the embedding space.
            hidden_dim (int): Dimension of the hidden layers.
            num_attention_layers (int): Number of attention layers.
            num_heads (int): Number of attention heads.
        """
        super().__init__(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )

        self.encoder = GraphDemandEncoder(
            depot_input_dim=depot_dim,
            node_input_dim=node_dim,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )

        # project [load, battery] (2 scalars) into the same space as load alone
        # The decoder expects a single scalar for load context.
        # We concatenate load and battery and project to 1 scalar.
        self.battery_proj = torch.nn.Linear(2, 1, bias=False)

    def forward(self, env, rollout=False) -> Tuple[float, float]:
        """
        Forward method of the DroneVRPModel.

        Args:
            env: DroneVRPEnv instance.
            rollout (bool): If True, act greedily. Defaults to False.

        Returns:
            Tuple[float, float]: accumulated loss and log probabilities.
        """
        done = False

        graph_state, load, battery = env.get_state()
        graph_state = torch.tensor(graph_state, dtype=torch.float, device=self.device)
        load = torch.tensor(load, dtype=torch.float, device=self.device)
        battery = torch.tensor(battery, dtype=torch.float, device=self.device)

        acc_loss = torch.zeros(size=(graph_state.shape[0],), device=self.device)
        acc_log_prob = torch.zeros(size=(graph_state.shape[0],), device=self.device)

        # encode graph once
        emb = self.encoder(
            x=graph_state[:, :, :3],
            depot_mask=graph_state[:, :, 3].bool()
        )

        while not done:
            # combine load and battery into single context scalar
            combined = torch.stack([load, battery], dim=1)  # (batch, 2)
            vehicle_state = self.battery_proj(combined).squeeze(-1)  # (batch,)

            actions, log_prob = self.decoder(
                node_embs=emb,
                mask=graph_state[:, :, -1],
                load=vehicle_state,
                rollout=rollout,
            )

            state, loss, done, _ = env.step(actions.cpu().numpy())

            acc_loss += torch.tensor(loss, dtype=torch.float, device=self.device)
            acc_log_prob += log_prob.squeeze().to(self.device)

            graph_state, load, battery = env.get_state()
            graph_state = torch.tensor(graph_state, dtype=torch.float, device=self.device)
            load = torch.tensor(load, dtype=torch.float, device=self.device)
            battery = torch.tensor(battery, dtype=torch.float, device=self.device)

        self.decoder.reset()

        return acc_loss, acc_log_prob


class DroneVRPAgent(TSPAgent):
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
        """
        DroneVRPAgent solves the drone VRP with payload and battery constraints.

        Args:
            depot_dim (int): Input dimension of a depot node.
            node_dim (int): Input dimension of a regular graph node.
            emb_dim (int): Embedding size.
            hidden_dim (int): Hidden layer size.
            num_attention_layers (int): Number of attention layers.
            num_heads (int): Number of attention heads.
            lr (float): Learning rate.
            csv_path (str): Path to save training logs.
            seed (int): Random seed.
        """
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

        self.model = DroneVRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model = DroneVRPModel(
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
