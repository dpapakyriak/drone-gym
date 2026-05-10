# Drone-GYM

An extension of [VRP-GYM](https://github.com/kevin-schumann/VRP-GYM) introducing 
range-constrained drone routing and sequential multi-drone fleet support, developed 
as a course project for EECS 6892 (Reinforcement Learning) at Columbia University.

## Base Environment

This repository builds directly on VRP-GYM by Kevin Schumann (Leibniz Universität 
Hannover), which implements the attention model of Kool et al. (2019) in an 
OpenAI Gymnasium-compatible interface. For installation instructions, environment 
setup, and base environment documentation, please refer to the 
[original repository](https://github.com/kevin-schumann/VRP-GYM).

## Extensions

Two extensions are introduced over the base VRP-GYM codebase.

### Range Constraint

The `DroneIRPEnv` environment augments the base IRP environment with a 
`current_range` state variable that decrements with each traversed edge, 
mirroring the existing `current_load` tracker. At every decision step, nodes 
that cannot be reached while preserving the ability to return to the depot within 
the remaining range budget are masked out, enforcing a hard feasibility constraint 
throughout the episode. Upon returning to the depot, both payload and range are 
fully replenished.

### Sequential Multi-Drone Dispatch

The environment supports fleets of k > 1 homogeneous drones via sequential dispatch. 
Once a drone exhausts its resources and returns to the depot, the next drone departs 
and the policy is applied afresh over the remaining unvisited nodes. This design 
deliberately avoids modifications to the underlying attention model architecture, 
preserving full compatibility with the trained policy.

## Installation

Please follow the installation instructions in the 
[original VRP-GYM repository](https://github.com/kevin-schumann/VRP-GYM). 
Once the base environment is set up, no additional dependencies are required 
to run the drone extensions.

## Usage

### Training a Single-Drone Model

```bash
python train_models.py --num_nodes 20 --max_range 3.0 --num_drones 1
```

### Training a Multi-Drone Model

```bash
python train_models.py --num_nodes 20 --max_range 3.0 --num_drones 3
```

### Reproducing Results

```bash
python irp_results.py        # IRP baseline (CVRP) results and training curves
python drone_results.py      # DroneIRP single-drone results
python heuristic.py          # Clarke-Wright savings heuristic benchmark
```

## Experiments

Models were trained and evaluated across the following configurations.

**Node sizes** — 20, 30, 40 demand nodes

**Range budgets** — R = 3.0, 4.0, 5.0 (Euclidean coordinate space)

**Fleet sizes** — k = 1, 3, 5, 10 drones

**Seeds** — 69, 123

**Epochs** — 851 per configuration


## Key Results

The trained attention model internalizes both payload and range constraints with 
negligible degradation in solution quality relative to the unconstrained CVRP 
baseline, while outperforming the Clarke-Wright savings heuristic by approximately 
10 to 15 percent across all problem sizes. Per-drone routing efficiency improves 
naturally as fleet size grows, suggesting that the sequential dispatch policy 
implicitly partitions demand nodes into compact clusters without any explicit 
coordination mechanism.

## Citation

If you use this code, please cite the original VRP-GYM repository and the 
foundational paper it is built upon.