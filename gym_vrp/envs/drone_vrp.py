from typing import Tuple, Union

import numpy as np

from .irp import IRPEnv
from .common import ObsType


class DroneVRPEnv(IRPEnv):
    """
    DroneVRPEnv extends IRPEnv to model a drone-based vehicle routing problem.
    In addition to the payload/capacity constraint from IRPEnv, the drone has
    a battery/range constraint: it can only travel a maximum distance (max_range)
    before it must return to the depot to recharge.

    A node is visitable only if:
        1. Its demand does not exceed the drone's current load (from IRPEnv)
        2. The drone can reach it AND return to depot within remaining battery

    State: Shape (batch_size, num_nodes, 5) — same as IRPEnv
        [x_coord, y_coord, demand, is_depot, visitable]

    Additional state returned: (load, battery) — both shape (batch_size,)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 32,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
        max_range: float = 3.0,
    ):
        """
        Args:
            num_nodes (int): Number of nodes in each graph. Defaults to 32.
            batch_size (int): Number of graphs. Defaults to 128.
            num_draw (int): Graphs to render. Defaults to 6.
            seed (int): Random seed. Defaults to 69.
            max_range (float): Maximum battery range of the drone. Defaults to 3.0.
        """
        self.max_range = max_range
        super().__init__(
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_draw=num_draw,
            seed=seed,
        )
        # battery initialized after super().__init__ which calls reset -> generate_graphs
        self.battery = np.ones(shape=(batch_size,)) * max_range

    def _get_node_positions(self):
        """Returns node positions as array of shape (batch_size, num_nodes, 2)."""
        return self.sampler.get_graph_positions()

    def _compute_distance(self, from_nodes, to_nodes):
        """
        Compute euclidean distance between from_nodes and to_nodes.
        Args:
            from_nodes: shape (batch_size,) — node indices
            to_nodes: shape (batch_size,) — node indices
        Returns:
            distances: shape (batch_size,)
        """
        positions = self._get_node_positions()  # (batch_size, num_nodes, 2)
        batch_idx = np.arange(self.batch_size)
        from_pos = positions[batch_idx, from_nodes.squeeze()]  # (batch_size, 2)
        to_pos = positions[batch_idx, to_nodes.squeeze()]      # (batch_size, 2)
        return np.linalg.norm(from_pos - to_pos, axis=1)       # (batch_size,)

    def step(self, actions: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Same as IRPEnv.step but also updates battery.
        Battery decreases by distance traveled each step.
        Battery resets to max_range when drone returns to depot.
        """
        assert (
            actions.shape[0] == self.batch_size
        ), "Number of actions must equal batch size."

        self.step_count += 1

        # compute distance traveled this step
        distances = self._compute_distance(self.current_location, actions)

        # visit nodes and update visited
        self.visited[np.arange(len(actions)), actions.T] = 1
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        self.sampler.visit_edges(traversed_edges)

        # update load (from IRPEnv)
        selected_demands = self.demands[
            np.arange(len(self.demands)), actions.T
        ].squeeze()
        self.load -= selected_demands
        self.load[np.where(actions == self.depots)[0]] = 1

        # update battery
        self.battery -= distances
        self.battery[np.where(actions == self.depots)[0]] = self.max_range

        self.current_location = np.array(actions)

        done = self.is_done()
        return (
            self.get_state(),
            -self.sampler.get_distances(traversed_edges),
            done,
            None,
        )

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns state tuple: (graph_state, load, battery)
        graph_state: shape (batch_size, num_nodes, 5)
        load: shape (batch_size,)
        battery: shape (batch_size,)
        """
        state = np.dstack(
            [
                self.sampler.get_graph_positions(),
                self.demands,
                np.zeros((self.batch_size, self.num_nodes)),
                self.generate_mask(),
            ]
        )
        state[np.arange(len(state)), self.depots.T, 3] = 1
        return (state, self.load, self.battery)

    def generate_mask(self):
        """
        Mask nodes that are:
        1. Already visited (from IRPEnv)
        2. Demand exceeds current load (from IRPEnv)
        3. Cannot be reached AND return to depot within remaining battery
        """
        positions = self._get_node_positions()  # (batch_size, num_nodes, 2)

        # depot positions: shape (batch_size, 2)
        batch_idx = np.arange(self.batch_size)
        depot_pos = positions[batch_idx, self.depots.squeeze()]  # (batch_size, 2)

        # current positions: shape (batch_size, 2)
        current_pos = positions[batch_idx, self.current_location.squeeze()]  # (batch_size, 2)

        # for each node: dist(current -> node) + dist(node -> depot)
        # shape: (batch_size, num_nodes)
        dist_to_node = np.linalg.norm(
            positions - current_pos[:, np.newaxis, :], axis=2
        )
        dist_node_to_depot = np.linalg.norm(
            positions - depot_pos[:, np.newaxis, :], axis=2
        )
        range_needed = dist_to_node + dist_node_to_depot  # (batch_size, num_nodes)

        # disallow staying at depot
        depot_graphs_idxs = np.where(self.current_location == self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1

        # allow visiting depot when not currently there
        depot_graphs_idxs_not = np.where(self.current_location != self.depots)[0]
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0

        # allow staying at depot if graph is solved
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        # start with visited mask
        mask = np.copy(self.visited)

        # mask nodes where demand exceeds load
        exceed_demand_idxs = ((self.demands - self.load[:, None, None]) > 0).squeeze()
        mask[exceed_demand_idxs] = 1

        # mask nodes where range is insufficient
        exceed_range = range_needed > self.battery[:, np.newaxis]
        mask[exceed_range] = 1

        # safety: always keep depot unmasked for fully solved graphs
        if len(done_graphs) > 0:
            mask[done_graphs, self.depots[done_graphs].squeeze()] = 0

        # safety: if all nodes masked (stuck), unmask depot
        all_masked = np.all(mask == 1, axis=1)
        stuck_graphs = np.where(all_masked)[0]
        if len(stuck_graphs) > 0:
            mask[stuck_graphs, self.depots[stuck_graphs].squeeze()] = 0

        return mask

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets environment including battery."""
        super().reset()
        self.battery = np.ones(shape=(self.batch_size,)) * self.max_range
        return self.get_state()
