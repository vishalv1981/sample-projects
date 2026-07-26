#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
profile=${1:-smoke}; profile_file="$PROJECT_DIR/experiments/profiles/$profile.env"; test -f "$profile_file" || die "Unknown profile: $profile"
override_pods=${SYNTHETIC_PODS:-}; . "$profile_file"; test -z "$override_pods" || SYNTHETIC_PODS=$override_pods; require_positive_integer SYNTHETIC_PODS "$SYNTHETIC_PODS"; test "$SYNTHETIC_PODS" -le "$POD_COUNT_LIMIT" || die 'Pod limit exceeded'
case "$profile" in medium|large) test "$CONFIRM_LARGE" = true || die "$profile requires --confirm-large";; esac
experiment_id="gpu-scale-$(date -u +%Y%m%dT%H%M%SZ)-$$"; experiment_id_valid "$experiment_id" || die 'Generated unsafe experiment ID'
result_dir="$RESULTS_DIR/$experiment_id"; mkdir -p "$result_dir"; printf '%s\n' "$experiment_id" > "$RUNTIME_DIR/latest-experiment"
start_epoch=$(date +%s); log "Impact estimate: $SYNTHETIC_PODS synthetic pods in batches of $BATCH_SIZE for $profile ($experiment_id)."
if test "$DRY_RUN" = true; then log "Dry run: would create $SYNTHETIC_PODS experiment-owned pods and collect results under $result_dir"; exit 0; fi
printf 'command=scale profile=%s experiment_id=%s batch_size=%s timeout_seconds=%s\n' "$profile" "$experiment_id" "$BATCH_SIZE" "$WAIT_TIMEOUT_SECONDS" > "$result_dir/command.log"
kube get --raw='/readyz?verbose' > "$result_dir/api-health-before.txt"
kube create namespace "$EXPERIMENT_NAMESPACE" --dry-run=client -o yaml | kube apply -f -
kube apply -f - <<EOF
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-scale-low
value: 100
globalDefault: false
description: Synthetic low-priority GPU workload
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-scale-high
value: 10000
globalDefault: false
description: Synthetic high-priority GPU workload
EOF

render_pod() {
  i=$1; scenario=heterogeneous; request=1; priority=gpu-scale-low; model=A100
  if test "$i" -le 5; then scenario=saturation; model=T4
  else
    case $((i%5)) in
      0) scenario=heterogeneous; model=H100;;
      1) scenario=heterogeneous; model=A100;;
      2) scenario=priority; priority=gpu-scale-high; model=T4;;
      3) scenario=fragmentation; request=2; model=A100;;
      4) scenario=fragmentation; request=4; model=H100;;
    esac
  fi
  cat <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: scale-$experiment_id-$i
  namespace: $EXPERIMENT_NAMESPACE
  labels:
    gpu-simulation.example/project: $PROJECT_LABEL_VALUE
    gpu-simulation.example/experiment-id: $experiment_id
    gpu-simulation.example/scenario: $scenario
spec:
  priorityClassName: $priority
  restartPolicy: Never
  nodeSelector:
    kwok.x-k8s.io/node: fake
    gpu-simulation.example/type: kwok
    gpu-simulation.example/gpu-model: $model
  tolerations:
    - key: kwok.x-k8s.io/node
      operator: Equal
      value: fake
      effect: NoSchedule
  containers:
    - name: simulated
      image: registry.k8s.io/pause:3.10.1
      resources:
        requests:
          nvidia.com/gpu: $request
        limits:
          nvidia.com/gpu: $request
---
EOF
}

while read -r first last; do manifest="$RUNTIME_DIR/pods-$first-$last.yaml"; : > "$manifest"; i=$first; while test "$i" -le "$last"; do render_pod "$i" >> "$manifest"; i=$((i+1)); done; if ! kube apply -f "$manifest" >> "$result_dir/command.log" 2>&1; then warn "Pod batch $first-$last failed; diagnostics preserved in $result_dir"; exit 1; fi; rm -f "$manifest"; done <<EOF
$(batch_ranges "$SYNTHETIC_PODS" "$BATCH_SIZE")
EOF

# Node-failure experiment: mark one project-owned KWOK node NotReady; never touch real nodes.
failure_node=$(kube get nodes -l 'kwok.x-k8s.io/node=fake,gpu-simulation.example/project=gpu-scheduling-scale-lab' -o jsonpath='{.items[0].metadata.name}')
test -n "$failure_node" && kube patch node "$failure_node" --subresource=status --type=merge -p '{"status":{"conditions":[{"type":"Ready","status":"False","reason":"SyntheticFailure","message":"Scale-lab simulated node failure","lastHeartbeatTime":"2020-01-01T00:00:00Z","lastTransitionTime":"2020-01-01T00:00:00Z"}]}}' >> "$result_dir/command.log"

sleep "$POLL_INTERVAL_SECONDS"
kube get pods -n "$EXPERIMENT_NAMESPACE" -l "gpu-simulation.example/experiment-id=$experiment_id" -o json > "$result_dir/pods.json"
kube get nodes -l 'kwok.x-k8s.io/node=fake,gpu-simulation.example/project=gpu-scheduling-scale-lab' -o json > "$result_dir/nodes.json"
jq -r '.items[] | [.metadata.name,.metadata.creationTimestamp,([.status.conditions[]? | select(.type=="PodScheduled") | .lastTransitionTime][0] // ""),(.spec.nodeName // ""),(.status.phase // "Pending"),([.status.conditions[]? | select(.type=="PodScheduled" and .status=="False") | .reason][0] // "")] | @csv' "$result_dir/pods.json" > "$result_dir/pod-scheduling.csv"
jq -r '.items[] | [.metadata.name,.metadata.labels["gpu-simulation.example/gpu-model"],.metadata.labels["gpu-simulation.example/node-pool"],.status.capacity["nvidia.com/gpu"],.status.allocatable["nvidia.com/gpu"],([.status.conditions[]? | select(.type=="Ready") | .status][0] // "Unknown")] | @csv' "$result_dir/nodes.json" > "$result_dir/node-summary.csv"
latencies="$result_dir/latencies.txt"; jq -r '.items[] | .metadata.creationTimestamp as $c | ([.status.conditions[]? | select(.type=="PodScheduled") | .lastTransitionTime][0] // empty) as $s | select($s != "") | (($s|fromdateiso8601)-($c|fromdateiso8601))' "$result_dir/pods.json" | sort -n > "$latencies"
scheduled=$(jq '[.items[] | select(.spec.nodeName != null)] | length' "$result_dir/pods.json"); running=$(jq '[.items[] | select(.status.phase=="Running")] | length' "$result_dir/pods.json"); pending=$(jq '[.items[] | select(.status.phase=="Pending")] | length' "$result_dir/pods.json"); failed=$(jq '[.items[] | select(.status.phase=="Failed")] | length' "$result_dir/pods.json"); ready_nodes=$(jq '[.items[] | select(any(.status.conditions[]?; .type=="Ready" and .status=="True"))] | length' "$result_dir/nodes.json")
test "$scheduled" -ge 1 || die 'No synthetic GPU pod scheduled onto a KWOK node; diagnostics preserved'
saturation_pending=$(jq '[.items[] | select(.metadata.labels["gpu-simulation.example/scenario"]=="saturation" and .status.phase=="Pending")] | length' "$result_dir/pods.json")
test "$saturation_pending" -ge 1 || warn 'Saturation overflow was not Pending at collection time; inspect pod-scheduling.csv and scheduler events.'
requested_gpus=$(jq '[.items[].spec.containers[].resources.requests["nvidia.com/gpu"] | tonumber] | add // 0' "$result_dir/pods.json")
allocated_gpus=$(jq '[.items[] | select(.spec.nodeName != null) | .spec.containers[].resources.requests["nvidia.com/gpu"] | tonumber] | add // 0' "$result_dir/pods.json")
total_gpu_capacity=$(jq '[.items[].status.allocatable["nvidia.com/gpu"] | tonumber] | add // 0' "$result_dir/nodes.json")
unallocated_gpus=$((total_gpu_capacity-allocated_gpus)); test "$unallocated_gpus" -ge 0 || unallocated_gpus=0
fragmentation_pending=$(jq '[.items[] | select(.metadata.labels["gpu-simulation.example/scenario"]=="fragmentation" and .status.phase=="Pending")] | length' "$result_dir/pods.json")
model_distribution=$(jq '[.items[] | select(.spec.nodeName != null) | .spec.nodeSelector["gpu-simulation.example/gpu-model"]] | group_by(.) | map({model:.[0],pods:length})' "$result_dir/pods.json")
end_epoch=$(date +%s); duration=$((end_epoch-start_epoch)); test "$duration" -gt 0 || duration=1; throughput=$(awk -v n="$scheduled" -v d="$duration" 'BEGIN{printf "%.3f",n/d}')
p50=$(percentile 50 < "$latencies"); p95=$(percentile 95 < "$latencies"); p99=$(percentile 99 < "$latencies")
jq -n --arg id "$experiment_id" --arg profile "$profile" --argjson requestedNodes "$KWOK_NODES" --argjson readyNodes "$ready_nodes" --argjson requestedPods "$SYNTHETIC_PODS" --argjson scheduled "$scheduled" --argjson running "$running" --argjson pending "$pending" --argjson failed "$failed" --argjson saturationPending "$saturation_pending" --argjson fragmentationPending "$fragmentation_pending" --argjson requestedGpus "$requested_gpus" --argjson allocatedGpus "$allocated_gpus" --argjson totalCapacity "$total_gpu_capacity" --argjson unallocatedGpus "$unallocated_gpus" --argjson modelDistribution "$model_distribution" --argjson duration "$duration" --arg throughput "$throughput" --arg p50 "$p50" --arg p95 "$p95" --arg p99 "$p99" '{experimentId:$id,profile:$profile,simulation:true,requestedKwokNodes:$requestedNodes,readyKwokNodes:$readyNodes,requestedSyntheticPods:$requestedPods,scheduledPods:$scheduled,simulatedRunningPods:$running,pendingPods:$pending,failedPods:$failed,saturationOverflowPendingPods:$saturationPending,fragmentationPendingPods:$fragmentationPending,requestedSimulatedGpus:$requestedGpus,allocatedSimulatedGpus:$allocatedGpus,totalSimulatedGpuCapacity:$totalCapacity,unallocatedSimulatedGpuCapacity:$unallocatedGpus,scheduledPodDistributionByGpuModel:$modelDistribution,durationSeconds:$duration,schedulingThroughputPodsPerSecond:($throughput|tonumber),schedulingLatencySeconds:{p50:($p50|tonumber),p95:($p95|tonumber),p99:($p99|tonumber)},notice:"Scheduler/control-plane simulation only; no containers or CUDA execute on KWOK nodes."}' > "$result_dir/summary.json"
{ printf 'host_os=%s\nhost_arch=%s\ndocker_arch=%s\nprofile=%s\n' "$(host_os)" "$(host_arch)" "$(docker_arch)" "$profile"; docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} memory={{.MemUsage}}' "$CLUSTER_NAME-control-plane" 2>/dev/null || true; } > "$result_dir/environment.txt"
kube get events -n "$EXPERIMENT_NAMESPACE" --field-selector reason=FailedScheduling -o json | jq -r '[.items[].message] | group_by(.) | map({reason:.[0],count:length})' > "$result_dir/unschedulable-reasons.json"
kube get --raw='/readyz?verbose' > "$result_dir/api-health-after.txt"; rm -f "$result_dir/pods.json" "$result_dir/nodes.json" "$latencies"
log "Results: $result_dir (simulated scheduling, not GPU performance)"
