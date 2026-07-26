# Running on an existing cloud VM

This project can run inside an already-created OCI Compute or AWS EC2 Ubuntu 22.04/24.04 instance. It does not provision, resize, stop, expose, or delete the VM and does not change cloud networking, IAM, security groups, firewalls, VCNs, or VPCs.

Recommended starting capacity is 8 vCPUs, 16 GiB RAM, and 50 GiB available disk for smoke/small. Medium and large need deliberate sizing and monitoring. AMD64 is the recommended full functional path. ARM64 works only when every pinned image reports ARM64 support.

Prerequisites:

- User-managed SSH access, for example `ssh <user>@<instance-address>`.
- Docker Engine installed and running using official Ubuntu instructions.
- Outbound HTTPS access to GitHub releases, `registry.k8s.io`, `ghcr.io`, Docker Hub, and Helm chart endpoints.
- Bash, curl, tar, awk, sed, grep, sort, and `sha256sum` or `shasum`.

Copy or clone the repository using your normal authentication, enter the project directory, then run:

```bash
./run.sh preflight
./run.sh bootstrap
./run.sh all --profile smoke
./run.sh status
./run.sh down
```

If Docker is missing, `--install-docker` prints official guidance but does not silently install packages. The scripts do not read `~/.aws`, `~/.oci`, SSH configuration, cloud metadata endpoints, or cloud credentials. All activity remains inside Docker and files under this project on the existing VM.
