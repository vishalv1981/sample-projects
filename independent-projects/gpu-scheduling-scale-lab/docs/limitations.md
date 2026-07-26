# Limitations

- No physical GPU, CUDA kernel, bandwidth, power, thermals, training, or inference execution is measured.
- KWOK nodes and pods exist in the Kubernetes API only; their containers never run.
- Fake GPU Operator behavior depends on upstream image architecture support. ARM64 must be checked, not assumed.
- Explicit AMD64 emulation can be slow and is not suitable for performance conclusions.
- Control-plane results depend on Docker resources, host load, Kubernetes/KWOK versions, batch size, and polling.
- Node-failure state is synthetic and does not reproduce all networking, storage, kubelet, or hardware failure behavior.
- Registry manifests showed AMD64 and ARM64 for the pinned kind, KWOK, and enabled Fake GPU images, but Docker Desktop was unavailable during implementation, so smoke integration is not reported as passed.
- Colima, non-Ubuntu Linux, private registries, proxies, and automated cloud provisioning are untested or out of scope.
