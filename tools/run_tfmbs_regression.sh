#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
LLVM_DIR="${LLVM_DIR:-$REPO_ROOT/../llvm-project/build/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-$REPO_ROOT/../llvm-project/build/lib/cmake/mlir}"

echo "Configuring MLIR plugin (build dir: $BUILD_DIR)"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR"

echo "Building tfmbs_plugin"
ninja -C "$BUILD_DIR" tfmbs_plugin

MLIR_OPT_PATH="${MLIR_OPT:-$REPO_ROOT/../llvm-project/build-shared/bin/mlir-opt}"
if [[ ! -x "$MLIR_OPT_PATH" ]]; then
  echo "mlir-opt not found at $MLIR_OPT_PATH" >&2
  exit 1
fi

if [[ -z "${TFMBS_PLUGIN:-}" ]]; then
  shopt -s nullglob
  plugin_candidates=("$BUILD_DIR"/libtfmbs_plugin.*)
  shopt -u nullglob
  if (( ${#plugin_candidates[@]} == 0 )); then
    echo "tfmbs_plugin build artifact not found in $BUILD_DIR" >&2
    exit 1
  fi
  TFMBS_PLUGIN="${plugin_candidates[0]}"
fi

echo "Running regression helper (mlir-opt: $MLIR_OPT_PATH, plugin: $TFMBS_PLUGIN)"
python3 "$REPO_ROOT/tests/mlir/run_tfmbs_to_linalg.py" \
  --mlir-opt "$MLIR_OPT_PATH" \
  --plugin "$TFMBS_PLUGIN"
