#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Restore the node prerequisites that a container restart does not re-apply.
#
#   ./kubernetes/node-recovery.sh
#
# `minikube start` configures all of this itself. The problem case is the
# VirtualBox guest rebooting while the container is stopped-down-by-the-host:
# the cluster is then brought back with `docker start minikube`, and the
# container entrypoint does not redo the following, so the cluster comes up
# "Running" but unusable:
#
#   1. Docker rewrites the container's /etc/hosts on start, dropping
#      `control-plane.minikube.internal`. kube-proxy's kubeconfig points at that
#      name, so kube-proxy cannot reach the apiserver and crashloops - Service
#      networking dies with it (CoreDNS never becomes Ready, every ClusterIP
#      times out, pods cannot resolve each other).
#   2. `br_netfilter` is not loaded, so `net.bridge.bridge-nf-call-iptables` does
#      not exist. Without it, pod traffic to a Service ClusterIP is never
#      DNAT'ed to an endpoint and connections hang instead of being refused.
#   3. kubelet is not started by the container entrypoint.
#
# Every step is idempotent, so this is safe to run on a healthy cluster; it exits
# non-zero when a prerequisite could not be restored. Run it on the machine that
# hosts the minikube container (the guest), as a user in the `docker` group - it
# needs no root on the host and never touches anything outside the container.
# ---------------------------------------------------------------------------
set -uo pipefail

CONTAINER="${MINIKUBE_CONTAINER:-minikube}"
log() { printf '==> %s\n' "$*"; }
detail() { printf '    %s\n' "$*"; }
fail=0
bad() {
  printf '    FAIL %s\n' "$*" >&2
  fail=1
}

command -v docker >/dev/null || {
  echo "error: docker not found in PATH" >&2
  exit 2
}

running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)"
if [[ "$running" != "true" ]]; then
  cat >&2 <<EOF
error: container '$CONTAINER' is not running.
  start the cluster the normal way:  minikube start --cpus=4 --memory=4096
  or, to reuse the existing container: docker start $CONTAINER && $0
EOF
  exit 2
fi

# ---------------------------------------------------------------------------
# 1. control-plane.minikube.internal -> apiserver
# ---------------------------------------------------------------------------
cp_ip=""
if command -v minikube >/dev/null; then
  cp_ip="$(minikube ip 2>/dev/null || true)"
fi
if [[ -z "$cp_ip" ]]; then
  cp_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
    "$CONTAINER" 2>/dev/null || true)"
fi

if [[ -z "$cp_ip" ]]; then
  bad "could not determine the control-plane IP (minikube ip and docker inspect both empty)"
else
  log "pinning control-plane.minikube.internal to $cp_ip"
  # Replace rather than append: a stale entry from a previous run must not win,
  # and duplicate lines make /etc/hosts ambiguous.
  #
  # /etc/hosts is a Docker bind mount, so `sed -i` cannot work here: it writes a
  # temp file and renames over the original, which fails with
  # "sed: cannot rename /etc/sedXXXXXX: Device or resource busy". Rewrite the file
  # in place instead (truncate + write is allowed on the mount).
  if docker exec -u root "$CONTAINER" sh -ec "
    { grep -v 'control-plane\.minikube\.internal' /etc/hosts || true; } > /tmp/hosts.recovery
    printf '%s control-plane.minikube.internal\n' '$cp_ip' >> /tmp/hosts.recovery
    cat /tmp/hosts.recovery > /etc/hosts
    rm -f /tmp/hosts.recovery
  "; then
    detail "wrote /etc/hosts entry"
  else
    bad "could not write the /etc/hosts entry"
  fi
fi

# ---------------------------------------------------------------------------
# 2. br_netfilter + bridge netfilter sysctls
# ---------------------------------------------------------------------------
log "loading br_netfilter and restoring the bridge sysctls"
docker exec -u root "$CONTAINER" sh -c '
  modprobe br_netfilter 2>/dev/null
  modprobe overlay 2>/dev/null
  sysctl -qw net.bridge.bridge-nf-call-iptables=1 \
             net.bridge.bridge-nf-call-ip6tables=1 \
             net.ipv4.ip_forward=1
' || bad "modprobe/sysctl step returned an error"

# ---------------------------------------------------------------------------
# 3. kubelet
# ---------------------------------------------------------------------------
state="$(docker exec -u root "$CONTAINER" systemctl is-active kubelet 2>/dev/null || true)"
if [[ "$state" == "active" ]]; then
  log "kubelet already active"
else
  log "starting kubelet inside the container (was: ${state:-unknown})"
  docker exec -u root "$CONTAINER" systemctl start kubelet \
    || bad "kubelet did not start"
fi

# ---------------------------------------------------------------------------
# Verification - a step that silently did nothing must not look successful.
# ---------------------------------------------------------------------------
log "verifying prerequisites"
resolved="$(docker exec "$CONTAINER" getent hosts control-plane.minikube.internal 2>/dev/null \
  | awk '{print $1; exit}')"
if [[ -n "$cp_ip" && "$resolved" == "$cp_ip" ]]; then
  detail "hosts: ok ($resolved)"
elif [[ -n "$resolved" ]]; then
  detail "hosts: stale ($resolved, expected $cp_ip)"
  bad "control-plane.minikube.internal resolves to $resolved"
else
  bad "control-plane.minikube.internal does not resolve"
fi

if docker exec -u root "$CONTAINER" sh -c \
  'test "$(cat /proc/sys/net/bridge/bridge-nf-call-iptables 2>/dev/null)" = 1'; then
  detail "br_netfilter: ok (bridge-nf-call-iptables=1)"
else
  bad "net.bridge.bridge-nf-call-iptables is not 1 (br_netfilter not loaded?)"
fi

state="$(docker exec -u root "$CONTAINER" systemctl is-active kubelet 2>/dev/null || true)"
if [[ "$state" == "active" ]]; then
  detail "kubelet: ok"
else
  bad "kubelet is '${state:-unknown}'"
fi

if command -v kubectl >/dev/null; then
  log "waiting for the apiserver (up to 60s)"
  healthy=false
  for _ in $(seq 1 10); do
    if kubectl get --raw /healthz --request-timeout=6s >/dev/null 2>&1; then
      healthy=true
      break
    fi
    sleep 6
  done
  if [[ "$healthy" == "true" ]]; then
    detail "apiserver: healthy"
  else
    bad "apiserver did not answer /healthz within 60s"
    detail "inspect: kubectl -n kube-system get pods; kubectl -n kube-system logs ds/kube-proxy"
  fi
else
  detail "kubectl not on PATH - skipped the apiserver check"
fi

if [[ "$fail" -ne 0 ]]; then
  printf '\nnode recovery FAILED - see the lines above\n' >&2
  exit 1
fi

printf '\nnode recovery OK - cluster prerequisites restored\n'
