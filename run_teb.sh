#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/outputs"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/teb_run_${TS}.log"

echo "[run_teb] Starting pipeline at $(date)" | tee "$LOG_FILE"
echo "[run_teb] Workspace: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "[run_teb] Log file: $LOG_FILE" | tee -a "$LOG_FILE"

cd "$ROOT_DIR"

# 1) Run experiment and save summary artifacts.
PYTHONUNBUFFERED=1 python3 teb.py 2>&1 | tee -a "$LOG_FILE"

# 2) Generate charts from saved summaries in outputs/.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg python3 charts.py 2>&1 | tee -a "$LOG_FILE"

echo "[run_teb] Completed at $(date)" | tee -a "$LOG_FILE"
echo "[run_teb] Outputs saved under: $OUT_DIR" | tee -a "$LOG_FILE"
