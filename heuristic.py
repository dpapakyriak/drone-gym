import numpy as np

def generate_graph(num_nodes):
    """Generate graph exactly as VRP-GYM does."""
    positions = np.random.rand(num_nodes, 2)
    depot = int(np.random.choice(num_nodes, size=1, replace=False)[0])
    C = 0.2449 * num_nodes + 26.12
    demands = np.random.uniform(low=1, high=10, size=(num_nodes,)) / C
    demands[depot] = 0
    return positions, demands, depot

def dist(positions, i, j):
    return np.linalg.norm(positions[i] - positions[j])

def clarke_wright(positions, demands, depot, capacity=1.0):
    """
    Clarke-Wright Savings Algorithm for CVRP.
    Single vehicle making multiple trips back to depot.
    """
    n = len(positions)
    customers = [i for i in range(n) if i != depot]

    # compute savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist(positions, depot, i) + dist(positions, depot, j) - dist(positions, i, j)
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # initialize: each customer is its own route
    routes = {i: [i] for i in customers}
    route_of = {i: i for i in customers}
    route_demand = {i: demands[i] for i in customers}

    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]

        if ri == rj:
            continue
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        if routes[ri][-1] != i or routes[rj][0] != j:
            continue

        new_route = routes[ri] + routes[rj]
        new_demand = route_demand[ri] + route_demand[rj]
        new_key = ri

        routes[new_key] = new_route
        route_demand[new_key] = new_demand
        for node in routes[rj]:
            route_of[node] = new_key
        del routes[rj]

    # compute total distance
    total = 0
    for r in routes.values():
        total += dist(positions, depot, r[0])
        for k in range(len(r) - 1):
            total += dist(positions, r[k], r[k+1])
        total += dist(positions, r[-1], depot)

    return total

def run_benchmark(num_nodes, seed, num_instances=256):
    np.random.seed(seed)
    costs = []
    for i in range(num_instances):
        positions, demands, depot = generate_graph(num_nodes)
        cost = clarke_wright(positions, demands, depot, capacity=1.0)
        costs.append(cost)
    return np.mean(costs)

if __name__ == '__main__':
    seeds = [69, 123]
    node_sizes = [20, 30, 40]

    print('=== Clarke-Wright CVRP Heuristic Benchmark ===\n')
    results = {}

    for num_nodes in node_sizes:
        costs_per_seed = []
        for seed in seeds:
            print(f'Running: {num_nodes} nodes, seed {seed}...')
            avg_cost = run_benchmark(num_nodes, seed, num_instances=256)
            costs_per_seed.append(avg_cost)
            print(f'  avg cost = {avg_cost:.4f}')

        results[num_nodes] = costs_per_seed
        print(f'{num_nodes} nodes overall avg: {np.mean(costs_per_seed):.4f}\n')

    print('\n=== Summary ===')
    print(f'{"Nodes":<10} {"Seed 69":<15} {"Seed 123":<15} {"Overall Avg":<15}')
    print('-' * 55)
    for num_nodes in node_sizes:
        s69, s123 = results[num_nodes]
        avg = np.mean(results[num_nodes])
        print(f'{num_nodes:<10} {s69:<15.4f} {s123:<15.4f} {avg:<15.4f}')

    print('\nNote: Clarke-Wright costs are positive (lower = better).')
    print('RL costs are negative — compare abs(RL cost) vs Clarke-Wright cost.')
