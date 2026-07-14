#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BENCH="$ROOT/llama.cpp/build/bin/llama-bench"
MODELS_DIR="$ROOT/models"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/$STAMP"
mkdir -p "$OUT"

[ -x "$BENCH" ] || { echo "llama-bench not found — run ./setup.sh first"; exit 1; }


PROMPTS="128,512,2048"      # prefill sizes: short chat, medium, long-context
GEN="128"                   # decode length
REPS=10                     # repetitions per test (llama-bench reports avg+stddev)
# M3 Pro 18GB = 11-core CPU (5 performance + 6 efficiency).
# Sweep: P-cores only, P+some E, all cores. Adjust if your chip differs
# (check: sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu)
CPU_THREADS="5 8 11"

{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "llama.cpp commit: $(cat "$ROOT/llama_commit.txt" 2>/dev/null || echo unknown)"
    echo "macos: $(sw_vers -productVersion)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string)"
    echo "p-cores: $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?')"
    echo "e-cores: $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo '?')"
    echo "memory_gb: $(( $(sysctl -n hw.memsize) / 1073741824 ))"
    echo "uptime: $(uptime)"
    echo "on_ac_power: $(pmset -g batt | head -1)"
    echo "reps: $REPS  prompts: $PROMPTS  gen: $GEN  cpu_threads: $CPU_THREADS"
} | tee "$OUT/env.txt"

echo
echo "NOTE: close other apps, stay on AC power, and don't touch the machine."
echo "      'caffeinate' will keep the Mac awake for the duration."
echo

run() {
    local model="$1" label="$2"; shift 2
    local file="$OUT/${label}.json"
    echo "==> $label"
    caffeinate -i "$BENCH" \
        -m "$model" \
        -p "$PROMPTS" -n "$GEN" -r "$REPS" \
        --progress -o json "$@" > "$file"
    echo "    saved $file"
}

shopt -s nullglob
MODELS=("$MODELS_DIR"/*.gguf)
[ ${#MODELS[@]} -gt 0 ] || { echo "No models in $MODELS_DIR"; exit 1; }

for MODEL in "${MODELS[@]}"; do
    NAME="$(basename "$MODEL" .gguf)"

    # GPU: all layers on Metal
    run "$MODEL" "${NAME}__metal" -ngl 99

    # CPU: NEON path, thread sweep
    for T in $CPU_THREADS; do
        run "$MODEL" "${NAME}__cpu_t${T}" -ngl 0 -t "$T"
    done
done

echo
echo "==> All runs complete. Analyze with:"
echo "    python3 analyze.py $OUT"
