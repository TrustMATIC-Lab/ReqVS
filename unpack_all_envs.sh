#!/usr/bin/env bash

set -e

ENVS_ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
ENVS_DIR="${ENVS_DIR:-$ENVS_ROOT/envs}"

# If envs directory exists, use it; otherwise use ENVS_ROOT
if [[ -d "$ENVS_DIR" ]]; then
  SEARCH_DIR="$ENVS_DIR"
else
  SEARCH_DIR="$ENVS_ROOT"
fi

cd "$ENVS_ROOT"

echo "===== Environment root: $ENVS_ROOT ====="
echo "===== Searching in: $SEARCH_DIR ====="

# Find all environments (both unpacked and packed)
all_envs=()
for d in "$SEARCH_DIR"/*/; do
  if [[ -d "$d" ]]; then
    name=$(basename "$d")
    if [[ -f "$d/bin/activate" ]] || [[ -f "$d/bin/conda-unpack" ]]; then
      all_envs+=("$name")
    fi
  fi
done

# Find environments that need to be unpacked
envs=()
for d in "$SEARCH_DIR"/*/; do
  if [[ -d "$d" ]]; then
    name=$(basename "$d")
    if [[ -f "$d/bin/conda-unpack" ]]; then
      envs+=("$name")
    fi
  fi
done

if [[ ${#envs[@]} -eq 0 ]]; then
  echo "No conda packed environments found (requires bin/conda-unpack)."
  echo ""
else
  echo "Found ${#envs[@]} environments to unpack: ${envs[*]}"
  echo ""

  failed=()
  for name in "${envs[@]}"; do
    env_path="$SEARCH_DIR/$name"
    echo "----- Unpacking: $name -----"
    if (
      set -e
      source "$env_path/bin/activate"
      conda-unpack
    ); then
      echo "----- $name unpack completed -----"
    else
      echo "----- $name unpack failed (may already be unpacked, can be ignored) -----"
      failed+=("$name")
    fi
    echo ""
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Environments that reported errors during unpack (may already be unpacked): ${failed[*]}"
    echo ""
  fi
fi

# List all available environments
if [[ ${#all_envs[@]} -gt 0 ]]; then
  echo "===== Available environments: ====="
  for i in "${!all_envs[@]}"; do
    env_name="${all_envs[$i]}"
    env_path="$SEARCH_DIR/$env_name"
    if [[ -f "$env_path/bin/activate" ]]; then
      echo "  [$((i+1))] $env_name"
    fi
  done
  echo ""
fi

echo "===== All environments processed ====="
echo ""
echo "Usage:"
echo "  Activate an environment: source $SEARCH_DIR/<env_name>/bin/activate"
echo "  Example: source $SEARCH_DIR/deepdtagen/bin/activate"
echo "  Call Python directly: $SEARCH_DIR/<env_name>/bin/python"
