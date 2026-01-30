#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$REPO_ROOT/bin"
SIM_BIN="$BIN_DIR/fabric_tb.vvp"
LOG_DIR="$REPO_ROOT/logs"
VVP="${VVP:-vvp}"

run_regression_helper() {
  if [[ -x "$REPO_ROOT/tools/run_tfmbs_regression.sh" ]]; then
    echo "Refreshing compiler regression fixtures..."
    "$REPO_ROOT/tools/run_tfmbs_regression.sh"
  fi
}

build_hw_sim() {
  echo "Building hardware simulation..."
  make -C "$REPO_ROOT" hw_sim
}

start_simulation() {
  mkdir -p "$LOG_DIR"
  echo "Starting Verilator fabric_tb simulation..."
  "$VVP" "$SIM_BIN" > "$LOG_DIR/fabric_tb.log" 2>&1 &
  SIM_PID=$!
  sleep 1
}

stop_simulation() {
  if [[ -n "${SIM_PID:-}" ]]; then
    echo "Stopping fabric simulation (PID $SIM_PID)"
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
}

collect_telemetry() {
  echo "Running DMA driver smoke test and capturing telemetry..."
  FABRIC_SHORT_CIRCUIT=1 TFMBS_DMA_RING=${TFMBS_DMA_RING:-256} "$BIN_DIR/test_dma_driver"
  python3 "$REPO_ROOT/tools/capture_dma_telemetry.py" \
    --libtfmbs "$BIN_DIR/libtfmbs_device${SHLIB_SUFFIX:-.so}" \
    --output "$LOG_DIR/adaptive_history_dma.json"
}

compare_telemetry() {
  echo "Comparing runtime telemetry to MLIR hints..."
  python3 "$REPO_ROOT/tools/adaptive_dashboard.py" \
    --runtime "$LOG_DIR/adaptive_history_dma.json" \
    --mlir "$REPO_ROOT/tests/mlir/torch_tfmbs.mlir" \
    --no-clash
}

if [[ -z "${SHLIB_SUFFIX:-}" ]]; then
  case "$(uname)" in
    Darwin) SHLIB_SUFFIX=".dylib" ;;
    *) SHLIB_SUFFIX=".so" ;;
  esac
fi

trap 'stop_simulation' EXIT

build_dma_driver() {
  echo "Building DMA driver test binary..."
  make -C "$REPO_ROOT" bin/test_dma_driver
}

run_regression_helper
build_hw_sim
build_dma_driver
start_simulation
collect_telemetry
compare_telemetry
