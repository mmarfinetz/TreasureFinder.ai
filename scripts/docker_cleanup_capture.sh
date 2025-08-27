#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Docker cleanup and disk-usage capture utility for macOS
# - Safe and non-interactive; guarded by flags
# - Captures baseline/after free space and sizes of Docker data dirs
# - Optionally performs a Docker Desktop factory-reset style cleanup

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_FILE="$REPO_DIR/.before_df_k.txt"
AFTER_FILE="$REPO_DIR/.after_df_k.txt"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="$REPO_DIR/cleanup_report_${TIMESTAMP}.log"

HOME_DIR="$HOME"
DOCKER_GROUP_DIR="$HOME_DIR/Library/Group Containers/group.com.docker"
DOCKER_CONTAINER_DIR="$HOME_DIR/Library/Containers/com.docker.docker"
DOCKER_DAEMON_DIR="$HOME_DIR/.docker"
DOCKER_RAW_FILE="$DOCKER_CONTAINER_DIR/Data/vms/0/data/Docker.raw"

is_baseline=false
is_cleanup=false
is_after=false
is_all=false
is_dry_run=false

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [--baseline] [--cleanup] [--after] [--all] [--dry-run]

  --baseline   Capture baseline disk usage (writes to .before_df_k.txt)
  --cleanup    Quit Docker and remove Docker Desktop data to reclaim space
  --after      Capture post-cleanup disk usage (writes to .after_df_k.txt)
  --all        Run baseline -> cleanup -> after in sequence
  --dry-run    Show what would be removed during cleanup without deleting
  -h, --help   Show this help message
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --baseline) is_baseline=true ;;
    --cleanup)  is_cleanup=true ;;
    --after)    is_after=true ;;
    --all)      is_all=true ;;
    --dry-run)  is_dry_run=true ;;
    -h|--help)  print_usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; print_usage; exit 1 ;;
  esac
done

# Default to baseline if no flags supplied
if ! $is_baseline && ! $is_cleanup && ! $is_after && ! $is_all; then
  is_baseline=true
fi

log() {
  printf "%s %s\n" "$(date '+%F %T')" "$*" | tee -a "$REPORT_FILE" >/dev/null
}

header() {
  printf "\n==== %s ====\n" "$*" | tee -a "$REPORT_FILE" >/dev/null
}

free_kb() {
  df -k / | awk 'NR==2{print $4}'
}

docker_df_safe() {
  if command -v docker >/dev/null 2>&1; then
    # Try a quick info call to see if the daemon is reachable
    if docker system info >/dev/null 2>&1; then
      docker system df 2>&1 | tee -a "$REPORT_FILE" >/dev/null || true
    else
      echo "Docker daemon not reachable; skipping 'docker system df'" | tee -a "$REPORT_FILE" >/dev/null
    fi
  else
    echo "docker CLI not installed; skipping 'docker system df'" | tee -a "$REPORT_FILE" >/dev/null
  fi
}

du_if_exists() {
  local path="$1"
  if [ -e "$path" ]; then
    du -sh "$path" 2>/dev/null | tee -a "$REPORT_FILE" >/dev/null || true
  else
    printf "missing %s\n" "$path" | tee -a "$REPORT_FILE" >/dev/null
  fi
}

capture_snapshot() {
  local label="$1" # BEFORE or AFTER
  header "$label snapshot"

  local kb
  kb="$(free_kb)"
  log "Free KB: $kb"

  header "df -h /"
  df -h / | tee -a "$REPORT_FILE" >/dev/null || true

  header "docker system df (best-effort)"
  docker_df_safe

  header "Docker desktop data directories"
  du_if_exists "$DOCKER_GROUP_DIR"
  du_if_exists "$DOCKER_CONTAINER_DIR"
  du_if_exists "$DOCKER_DAEMON_DIR"
  du_if_exists "$DOCKER_RAW_FILE"

  if [ "$label" = "BEFORE" ]; then
    echo "$kb" > "$BASELINE_FILE"
    printf "Before free KB: %s\n" "$kb"
  else
    echo "$kb" > "$AFTER_FILE"
    printf "After free KB: %s\n" "$kb"
    if [ -f "$BASELINE_FILE" ]; then
      local before_k
      before_k="$(cat "$BASELINE_FILE" 2>/dev/null || echo 0)"
      if [[ "$before_k" =~ ^[0-9]+$ ]]; then
        local delta_k
        delta_k=$(( kb - before_k ))
        log "Delta free KB (AFTER - BEFORE): $delta_k"
        printf "Delta free KB (AFTER - BEFORE): %s\n" "$delta_k"
      else
        log "Baseline file was invalid; cannot compute delta"
      fi
    fi
  fi
}

perform_cleanup() {
  header "Quit Docker Desktop if running"
  osascript -e 'quit app "Docker"' >/dev/null 2>&1 || true
  sleep 1
  pkill -9 -f 'Docker Desktop|com.docker|vpnkit|qemu|dockerd' >/dev/null 2>&1 || true

  header "Remove stale sockets"
  if $is_dry_run; then
    echo "DRY-RUN: rm -f $DOCKER_DAEMON_DIR/run/docker.sock" | tee -a "$REPORT_FILE" >/dev/null
  else
    rm -f "$DOCKER_DAEMON_DIR/run/docker.sock" 2>/dev/null || true
  fi

  header "Remove Docker Desktop data (factory reset)"
  for target in \
    "$DOCKER_GROUP_DIR" \
    "$DOCKER_DAEMON_DIR" \
    "$DOCKER_RAW_FILE" \
    "$DOCKER_CONTAINER_DIR" \
  ; do
    if $is_dry_run; then
      echo "DRY-RUN: rm -rf $target" | tee -a "$REPORT_FILE" >/dev/null
    else
      rm -rf "$target" 2>/dev/null || true
    fi
  done
}

# Execution flow
if $is_all; then
  capture_snapshot "BEFORE"
  perform_cleanup
  capture_snapshot "AFTER"
else
  $is_baseline && capture_snapshot "BEFORE"
  $is_cleanup && perform_cleanup
  $is_after && capture_snapshot "AFTER"
fi

header "Done"
log "Report saved to $REPORT_FILE"


