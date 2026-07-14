
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA="$ROOT/llama.cpp"
BIN="$LLAMA/build/bin"
MODELS_DIR="${1:?usage: ./profile_prefill.sh <models_dir>}"
OUT="$ROOT/profiles/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

THREADS=5         
PROMPT_TOKENS=512 

echo "==> Output: $OUT"
echo "==> Checking build options for repack support"
(cd "$LLAMA" && cmake -B build -LH 2>/dev/null | grep -iE "repack|aarch64" || true) \
    | tee "$OUT/build_options.txt"

echo "==> Dumping tensor types per model"
for MODEL in "$MODELS_DIR"/*.gguf; do
    NAME="$(basename "$MODEL" .gguf)"
    if command -v gguf-dump >/dev/null; then
        gguf-dump --no-tensors "$MODEL" > "$OUT/meta_${NAME}.txt" 2>/dev/null || true
        # tensor list with types; summarize type counts by layer-role
        gguf-dump "$MODEL" 2>/dev/null | grep -E "^\s+[0-9]+:" \
            > "$OUT/tensors_${NAME}.txt" || true
        awk '{print $NF, $(NF-1)}' "$OUT/tensors_${NAME}.txt" 2>/dev/null \
            | sort | uniq -c | sort -rn > "$OUT/tensor_type_summary_${NAME}.txt" || true
    else
        echo "gguf-dump not found (pip install gguf) — skipping tensor dump"
        break
    fi
done


echo "==> Capturing system_info / load-time repack messages"
for MODEL in "$MODELS_DIR"/*Q4_0*.gguf "$MODELS_DIR"/*F16*.gguf; do
    [ -f "$MODEL" ] || continue
    NAME="$(basename "$MODEL" .gguf)"
    "$BIN/llama-cli" -m "$MODEL" -ngl 0 -t $THREADS -n 1 -p "hi" --no-warmup -v 2>&1 \
        | grep -iE "system_info|repack|aarch64|CPU :" \
        > "$OUT/sysinfo_${NAME}.txt" || true
    echo "    $NAME:"
    sed 's/^/      /' "$OUT/sysinfo_${NAME}.txt" | head -5
done


echo "==> Sampling prefill call graphs (30s each)"
for MODEL in "$MODELS_DIR"/*F16*.gguf "$MODELS_DIR"/*Q4_0*.gguf "$MODELS_DIR"/*Q4_K_M*.gguf; do
    [ -f "$MODEL" ] || continue
    NAME="$(basename "$MODEL" .gguf)"
    case "$NAME" in *hip*) continue;; esac   # skip the repacked variant here
    echo "    profiling $NAME"
    "$BIN/llama-bench" -m "$MODEL" -p $PROMPT_TOKENS -n 0 -r 50 \
        -ngl 0 -t $THREADS -o json > "$OUT/bench_${NAME}.json" &
    PID=$!
    sleep 5                          # skip model load
    sample "$PID" 30 -file "$OUT/sample_${NAME}.txt" >/dev/null 2>&1 || true
    wait "$PID" || true
done


echo "==> Building and running test-backend-ops (MUL_MAT perf)"
(cd "$LLAMA" && cmake --build build --target test-backend-ops -j 8 >/dev/null)
"$BIN/test-backend-ops" perf -o MUL_MAT -b CPU > "$OUT/backend_ops_perf.txt" 2>&1 || true
grep -E "type_a=(f16|q4_0|q4_K|q8_0)" "$OUT/backend_ops_perf.txt" \
    > "$OUT/backend_ops_perf_filtered.txt" || true

echo
echo " Read in this order:"
echo "    1. sysinfo_*         — is REPACK active for Q4_0?"
echo "    2. tensor_type_summary_* — what's actually quantized in each GGUF?"
echo "    3. sample_*          — where do prefill cycles go, F16 vs Q4?"
echo "    4. backend_ops_perf_filtered.txt — raw kernel speed, type vs type"
