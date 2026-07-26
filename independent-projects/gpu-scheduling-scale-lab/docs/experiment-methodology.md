# Experiment methodology

One run owns resources through `gpu-simulation.example/experiment-id=<unique-id>` and writes to one result directory. Batches prevent shell argument and API request-size limits.

Scenarios are interleaved deterministically:

1. **Heterogeneous scheduling:** pods select T4, A100, or H100 labels.
2. **GPU saturation:** one-GPU pods consume advertised capacity; excess remains Pending.
3. **Priority and preemption:** low/high PriorityClasses record standard-scheduler decisions.
4. **Node failure:** one project-owned KWOK node transitions away from Ready.
5. **Fragmentation:** one-, two-, and four-GPU requests expose stranded simulated capacity.

Measurements count requested/Ready nodes, requested/scheduled/simulated-Running/Pending/Failed pods, duration, throughput, latency percentiles, model/pool distribution, requested/allocated/total/unallocated GPU values, fragmentation-pending pods, unschedulable reasons, API health, and best-effort kind control-plane Docker statistics. Unallocated capacity is an upper-bound fragmentation indicator: it also includes intentionally idle GPUs and must be interpreted together with fragmentation-pending pods and `node-summary.csv`.

Latency uses pod creation time and the first `PodScheduled` condition transition. Clock resolution, API admission, scheduler queueing, binding, watch propagation and KWOK stage timing are included. Results are scheduler/control-plane observations only.
