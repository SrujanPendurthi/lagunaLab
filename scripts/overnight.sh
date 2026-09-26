#!/usr/bin/env bash
# Overnight GPU work queue (Part 2 controller ruling: Part 2 before S4).
#
#   caffeinate -dims bash scripts/overnight.sh
#
# Runs s1, the Part-2 prune_sweep chain, and s4 sequentially and unattended.
# s1/Part-2/s4 are independent -- a failure in one does not stop the others.
# Within the Part-2 chain, a failure stops the rest of that chain only.
#
# Safety: refuses to start if a heavy model process is already running;
# before every step, halts all remaining steps if results/overnight/STOP
# exists, if swap use is over budget, or if swap can't be read (fails
# closed). Per-step stdout+stderr -> results/overnight/<step>.log; one
# start/end/skip line per step -> results/overnight/queue.log.
set -u

cd "$(dirname "$0")/.." || exit 1

LOG_DIR="results/overnight"
QUEUE_LOG="$LOG_DIR/queue.log"
STOP_FILE="$LOG_DIR/STOP"
STATS_FILE="results/expert_stats.npz"
MAX_SWAP_GB=10
HEAVY_PATTERN='[o]llama serve|[l]lama-server|[s]cripts.serve_streaming'

mkdir -p "$LOG_DIR"

qlog() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >>"$QUEUE_LOG"; }

# Refuse to start at all if something heavy is already running (bracket
# trick in HEAVY_PATTERN keeps this pgrep from matching itself).
if pgrep -f "$HEAVY_PATTERN" >/dev/null 2>&1; then
    qlog "ABORT: heavy model process already running (pgrep -f '$HEAVY_PATTERN')"
    exit 1
fi

HALTED=0
HALT_REASON=""

# Idempotent: once halted, stays halted for the rest of the run. Otherwise
# re-checks the STOP file and current swap use (both can change between
# steps overnight).
check_gate() {
    [ "$HALTED" -eq 1 ] && return
    if [ -e "$STOP_FILE" ]; then
        HALTED=1
        HALT_REASON="STOP file present ($STOP_FILE)"
        return
    fi
    local raw re used unit gb
    raw="$(sysctl vm.swapusage 2>/dev/null)"
    re='used = ([0-9.]+)([MG])'
    if [[ "$raw" =~ $re ]]; then
        used="${BASH_REMATCH[1]}"
        unit="${BASH_REMATCH[2]}"
        if [ "$unit" = "G" ]; then
            gb="$used"
        else
            gb="$(awk -v m="$used" 'BEGIN { printf "%.4f", m / 1024 }')"
        fi
        if awk -v g="$gb" -v max="$MAX_SWAP_GB" 'BEGIN { exit !(g > max) }'; then
            HALTED=1
            HALT_REASON="swap used ${gb}G > ${MAX_SWAP_GB}G budget"
        fi
    else
        HALTED=1
        HALT_REASON="could not parse 'sysctl vm.swapusage' output, failing closed"
    fi
}

# run_step NAME CMD...  Returns CMD's exit code, or 1 if skipped.
run_step() {
    local name="$1"; shift
    check_gate
    if [ "$HALTED" -eq 1 ]; then
        qlog "SKIP $name: $HALT_REASON"
        return 1
    fi
    qlog "START $name: $*"
    "$@" >"$LOG_DIR/$name.log" 2>&1
    local code=$?
    qlog "END $name exit=$code"
    return "$code"
}

skip_step() {
    qlog "SKIP $1: $2"
}

# 1. s1 -- independent
run_step s1 python3 -m scripts.diag_dflash --target both --skip-existing

# 2-4. Part-2 prune_sweep chain -- independent of s1/s4; a failure here
# stops the rest of this chain only.
calib_ok=0
check_gate
if [ "$HALTED" -eq 1 ]; then
    skip_step p2_calib "$HALT_REASON"
elif [ -e "$STATS_FILE" ]; then
    skip_step p2_calib "$STATS_FILE already exists"
    calib_ok=1
elif run_step p2_calib python3 -m scripts.prune_sweep calibrate; then
    calib_ok=1
fi

refs_ok=0
if [ "$calib_ok" -eq 1 ]; then
    run_step p2_refs python3 -m scripts.prune_sweep refs && refs_ok=1
else
    skip_step p2_refs "prune_sweep calibrate did not succeed"
fi

if [ "$refs_ok" -eq 1 ]; then
    run_step p2_sweep python3 -m scripts.prune_sweep sweep
else
    skip_step p2_sweep "prune_sweep refs did not succeed"
fi

# 5. s4 -- independent
run_step s4 python3 -m scripts.bench_lookup --skip-existing

exit 0
