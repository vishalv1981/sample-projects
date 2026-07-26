# Architecture and trust boundaries

The kind control plane and one real worker are containers managed by Docker. Only the worker labeled `run.ai/simulated-gpu-node-pool=default` is eligible for Run:ai Fake GPU Operator components and functional pods.

KWOK is configured with `manageAllNodes: false` and the annotation selector `kwok.x-k8s.io/node=fake`. Every synthetic node has that annotation and label, `gpu-simulation.example/type=kwok`, a project label, a GPU-model/node-pool label, and the taint `kwok.x-k8s.io/node=fake:NoSchedule`. Synthetic pods both tolerate and select this boundary. Real kind nodes are kubelet-managed and never marked for KWOK.

T4 nodes advertise four, A100 eight, and H100 eight `nvidia.com/gpu` resources directly through the Node status subresource. KWOK controllers simulate readiness and pod lifecycle. There is no kubelet, device plugin, container runtime, NVIDIA driver, CUDA execution, or application process on a KWOK node.

The functional path is separate: the public Run:ai chart advertises fake GPU resources on the real worker. A validation pod requests one resource and is checked for placement on that worker. Any fake `nvidia-smi` output describes the operator simulation only.
