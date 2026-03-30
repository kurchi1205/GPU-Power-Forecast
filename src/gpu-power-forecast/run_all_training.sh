#!/usr/bin/env bash
# Runs all baseline and LSTM training scripts sequentially.
# Results are logged to training_results.log with timestamps.

set -euo pipefail

LOG="training_results.log"
PYTHON="${PYTHON:-python3}"

log() {
    echo "$1" | tee -a "$LOG"
}

divider() {
    log "================================================================"
}

run_script() {
    local name="$1"
    local script="$2"

    divider
    log ">>> [$name]  started at $(date '+%Y-%m-%d %H:%M:%S')"
    divider

    if $PYTHON "$script" 2>&1 | tee -a "$LOG"; then
        log ">>> [$name]  finished at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        log ">>> [$name]  FAILED at $(date '+%Y-%m-%d %H:%M:%S')"
        exit 1
    fi

    echo "" | tee -a "$LOG"
}

# ── init log ──────────────────────────────────────────────────────────────────
echo "" > "$LOG"
log "GPU Power Forecast — Full Training Run"
log "Started : $(date '+%Y-%m-%d %H:%M:%S')"
log "Python  : $($PYTHON --version 2>&1)"
log "Device  : $(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")"
divider

# ── scripts in ablation order ─────────────────────────────────────────────────
run_script "Naive Baseline"         run_naive.py
run_script "InstantaneousMLP"       train_instantaneous_mlp.py
run_script "WindowedLinear + MLP"   train_windowed_mlp.py
run_script "PowerGRU + PowerRNN"    train_rnn_baselines.py
run_script "PowerLSTM"              train_lstm.py

# ── done ──────────────────────────────────────────────────────────────────────
divider
log "All runs completed at $(date '+%Y-%m-%d %H:%M:%S')"
log "Full log saved to: $LOG"
divider
