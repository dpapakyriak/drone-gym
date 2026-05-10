from typing import Tuple, Union

import numpy as np

from .drone_vrp import DroneVRPEnv
from .common import ObsType


class MultiDroneVRPEnv(DroneVRPEnv):
    """
    MultiDroneVRPEnv extends DroneVRPEnv to support multiple drones.
    Each drone has its own load, battery, and current location.
    A node is visited if ANY drone has visited it.
    Each drone gets its own mask based on its individual constraints.

    State returns:
        graph_state: (batch_size, num_nodes, 5)
        load: (batch_size, num_drones)
        battery: (batch_size, num_drones)
        drone_masks: (batch_size, num_drones, num_nodes)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 32,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
        max_range: float = 3.0,
        num_drones: int = 3,
    ):
        self.num_drones = num_drones
        super().__init__(
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_draw=num_draw,
            seed=seed,
            max_range=max_range,
        )

    def _init_drone_states(self):
        """Initialize per-drone state vectors."""
        self.current_location = np.repeat(self.depots, self.num_drones, axis=1)
        self.load = np.ones(shape=(self.batch_size, self.num_drones))
        self.battery = np.ones(shape=(self.batch_size, self.num_drones)) * self.max_range

    def generate_graphs(self):
        """Generate graphs and initialize multi-drone state."""
        from ..graph.vrp_network import VRPNetwork

        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            num_depots=1,
            plot_demand=True,
        )
        self.depots = self.sampler.get_depots()
        self.demands = self.sampler.get_demands()
        self._init_drone_states()

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.step_count = 0
        self.generate_graphs()
        return self.get_state()

    def generate_drone_masks(self):
        """
        Generate per-drone masks of shape (batch_size, num_drones, num_nodes).
        For each drone, a node is masked if:
        1. Already visited by any drone
        2. Demand exceeds this drone's load
        3. Range needed exceeds this drone's battery
        """
        positions = self._get_node_positions()  # (batch_size, num_nodes, 2)
        batch_idx = np.arange(self.batch_size)
        depot_pos = positions[batch_idx, self.depots.squeeze()]  # (batch_size, 2)

        # distance from each node to depot: (batch_size, num_nodes)
        dist_node_to_depot = np.linalg.norm(
            positions - depot_pos[:, np.newaxis, :], axis=2
        )

        # base visited mask
        base_visited = np.copy(self.visited)

        # solved graphs always keep depot unmasked
        done_graphs = np.where(np.all(self.visited == 1, axis=1))[0]

        drone_masks = np.zeros((self.batch_size, self.num_drones, self.num_nodes))

        for d in range(self.num_drones):
            mask = np.copy(base_visited)

            # depot logic: disallow depot if this drone is already there
            at_depot = np.where(
                self.current_location[:, d] == self.depots.squeeze()
            )[0]
            mask[at_depot, self.depots[at_depot].squeeze()] = 1

            # allow depot when not there
            not_at_depot = np.where(
                self.current_location[:, d] != self.depots.squeeze()
            )[0]
            mask[not_at_depot, self.depots[not_at_depot].squeeze()] = 0

            # allow depot for solved graphs
            if len(done_graphs) > 0:
                mask[done_graphs, self.depots[done_graphs].squeeze()] = 0

            # mask demand constraint
            exceed_demand = (
                (self.demands - self.load[:, d, None, None]) > 0
            ).squeeze()
            mask[exceed_demand] = 1

            # mask range constraint
            current_pos = positions[batch_idx, self.current_location[:, d]]
            dist_to_node = np.linalg.norm(
                positions - current_pos[:, np.newaxis, :], axis=2
            )
            range_needed = dist_to_node + dist_node_to_depot
            exceed_range = range_needed > self.battery[:, d, np.newaxis]
            mask[exceed_range] = 1

            # safety: if all masked, unmask depot
            all_masked = np.all(mask == 1, axis=1)
            stuck = np.where(all_masked)[0]
            if len(stuck) > 0:
                mask[stuck, self.depots[stuck].squeeze()] = 0

            drone_masks[:, d, :] = mask

        return drone_masks

    def generate_mask(self):
        """Returns shared mask for compatibility — union of all drone masks."""
        drone_masks = self.generate_drone_masks()
        # a node is unmasked if ANY drone can visit it
        return np.min(drone_masks, axis=1)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            graph_state: (batch_size, num_nodes, 5)
            load: (batch_size, num_drones)
            battery: (batch_size, num_drones)
            drone_masks: (batch_size, num_drones, num_nodes)
        """
        drone_masks = self.generate_drone_masks()
        shared_mask = np.min(drone_masks, axis=1)

        state = np.dstack([
            self.sampler.get_graph_positions(),
            self.demands,
            np.zeros((self.batch_size, self.num_nodes)),
            shared_mask,
        ])
        state[np.arange(len(state)), self.depots.T, 3] = 1

        return (state, self.load, self.battery, drone_masks)

    def step(self, actions: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Args:
            actions: shape (batch_size, num_drones)
        """
        assert actions.shape == (self.batch_size, self.num_drones), \
            f"Actions must be shape ({self.batch_size}, {self.num_drones})"

        self.step_count += 1
        total_distance = np.zeros(self.batch_size)

        for d in range(self.num_drones):
            drone_actions = actions[:, d:d+1]  # (batch_size, 1)

            distances = self._compute_distance(
                self.current_location[:, d:d+1], drone_actions
            )
            total_distance += distances

            self.visited[np.arange(self.batch_size), drone_actions.squeeze()] = 1

            traversed = np.hstack([
                self.current_location[:, d:d+1], drone_actions
            ]).astype(int)
            self.sampler.visit_edges(traversed)

            selected_demands = self.demands[
                np.arange(self.batch_size), drone_actions.squeeze()
            ].squeeze()
            self.load[:, d] -= selected_demands
            depot_returns = np.where(
                drone_actions.squeeze() == self.depots.squeeze()
            )[0]
            self.load[depot_returns, d] = 1.0
            self.battery[:, d] -= distances
            self.battery[depot_returns, d] = self.max_range

            self.current_location[:, d] = drone_actions.squeeze()

        done = self.is_done()
        return (self.get_state(), -total_distance, done, None)

    def is_done(self):
        return np.all(self.visited == 1)