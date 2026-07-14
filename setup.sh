#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$ROOT/llama.cpp"
MODELS_DIR="$ROOT/models"
JOBS="$(sysctl -n hw.ncpu)"

echo "==> Checking prerequisites"
command -v git   >/dev/null || { echo "git not found";   exit 1; }
command -v cmake >/dev/null || { echo "cmake not found (brew install cmake)"; exit 1; }
if ! command -v hf >/dev/null && ! command -v huggingface-cli >/dev/null; then
    echo "hf CLI not found. Install with: pip install -U huggingface_hub"
    exit 1
fi
HF_CLI="$(command -v hf || command -v huggingface-cli)"


if [ ! -d "$LLAMA_DIR" ]; then
    echo "==> Cloning llama.cpp"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git pull --ff-only || true
COMMIT="$(git rev-parse --short HEAD)"
echo "==> llama.cpp at commit $COMMIT"

echo "==> Building (Release, Metal on, Accelerate off for honest CPU numbers)"
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DLLAMA_CURL=ON
cmake --build build --config Release -j "$JOBS" --target llama-bench llama-cli llama-perplexity

echo "$COMMIT" > "$ROOT/llama_commit.txt"


mkdir -p "$MODELS_DIR"
REPO="LiquidAI/LFM2-1.2B-GGUF"

echo "==> Downloading LFM2-1.2B GGUFs from $REPO"
for PATTERN in "*F16*.gguf" "*Q8_0*.gguf" "*Q5_K_M*.gguf" "*Q4_K_M*.gguf" "*Q4_0*.gguf"; do
    "$HF_CLI" download "$REPO" --include "$PATTERN" --local-dir "$MODELS_DIR" || \
        echo "    (no file matching $PATTERN in $REPO — check repo file list manually)"
done

echo
echo "==> Done. Models in $MODELS_DIR:"
ls -lh "$MODELS_DIR" | grep -i gguf || echo "    NONE — check download step"
echo
echo "Next: ./run_bench.sh"
