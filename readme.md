# Liquid edge inference: finding a silent 2.4x TTFT loss on Apple Silicon

Performance investigation of [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF)
inference on Apple Silicon (M3 Pro, 18 GB) via llama.cpp — from a rigorous
baseline, through a wrong hypothesis, to a root cause, an upstream report,
and a tool that detects the whole class of problem automatically.

**TL;DR:** llama.cpp's default CPU weight repacking on Apple Silicon locks
quantized weights into a GPU-unreadable layout, silently disabling Metal
op-offload of prompt processing. At `-ngl 0`, this costs **2.4-2.7x
time-to-first-token** in exchange for a ~25% decode gain — on every
quantized model, on every Mac. Nothing warns you.

## The numbers

Apple M3 Pro, llama.cpp b9993, `-ngl 0 -t 5`, pp512/tg128, median of 10:

| model | config | prefill (t/s) | decode (t/s) | 100 ms TTFT fits |
|---|---|---:|---:|---:|
| LFM2-1.2B Q4_0 | default (repack ON) | 620 | 155 | 62 tokens |
| LFM2-1.2B Q4_0 | repack OFF | **1536** | 128 | **153 tokens** |
| Llama-3.2-1B Q4_0 | default (repack ON) | 618 | 137 | 62 tokens |
| Llama-3.2-1B Q4_0 | repack OFF | **1641** | 114 | 164 tokens |

Reference: full GPU offload (`-ngl 99`) = 1967 pp / 158 tg. The finding
only affects CPU-constrained inference — which on phones/tablets (thermal
throttling, GPU contention, battery policies) is the normal case, not the
corner case.

## What we detected

**The mechanism.** llama.cpp has two independently excellent optimizations
that compose badly:

1. **CPU repack** (default ON): at load time, quantized weights are
   rewritten into an ARM-interleaved layout (`CPU_REPACK` buffer type) so
   NEON/i8mm kernels run fast. Measured value: ~3.7x CPU prompt processing
   vs plain layout, +20-25% decode. Genuinely good.
2. **Op-offload**: on unified-memory Macs, large-batch matmuls (prompt
   processing) are offloaded to the Metal GPU even at `-ngl 0`, because the
   GPU can read CPU-resident weights for free. Measured value: ~9x prompt
   processing (1536 vs 167 t/s with offload disabled). Even better.

The collision: the repacked layout is a private CPU-kernel format the Metal
backend cannot read, so repacked tensors are ineligible for op-offload.
The load-time decision (repack) forecloses the run-time one (offload), and
the default silently picks the smaller win. The Pareto-optimal
configuration — GPU prefill + repacked-CPU decode — is structurally
unreachable because weights exist in exactly one layout.

**How it was found.** The initial baseline showed quantized prefill
2.5-3x *slower* than F16 on "CPU" — which sent us down the wrong path
(quantized kernel quality, LFM2's conv blocks) until a sanity check showed
the F16 number (1437 t/s) exceeded what five P-cores can physically
deliver. The F16 run was secretly using the GPU via op-offload; the Q4 run
couldn't. Profiling (`sample` call graphs), per-tensor buffer logs, ggml
micro-benchmarks, and a build-flag A/B on two model architectures confirmed
the mechanism. Full lab notes in [`docs/`](docs/).

Secondary findings along the way, all measured:
- **E-core oversubscription**: on 5P+6E, `-t 11` decode is 40-45% slower
  and far noisier than `-t 5` — barrier-synchronized kernels run at the
  pace of the slowest core.
- **Decode is bandwidth-bound as expected**: best decode = 73% of the
  M3 Pro's ~150 GB/s roofline; further decode gains require smaller quants,
  not faster kernels.
- `-ot "blk\..*=CPU"` is not a workaround (641 t/s — repack-level).
- `--no-repack` exists in llama-cli/server but **not in llama-bench**, so
  the trade-off cannot even be measured with stock tooling.

## What we're fixing

1. **Upstream visibility**: a llama.cpp issue with the full evidence
   ([draft](docs/github-issue-draft.md)) — quantifying the Metal-side
   interaction (the CUDA analogue was reported in
   [ggml-org/llama.cpp#12237](https://github.com/ggml-org/llama.cpp/issues/12237)).
2. **Tooling gap**: a PR adding `--no-repack` to llama-bench so the
   trade-off is measurable without rebuilding.
3. **The class of problem**: [`edge_audit.py`](edge_audit.py) — a
   "lighthouse for local LLMs" that runs a device through a short
   llama-bench matrix and detects this pathology (plus thread
   oversubscription, bandwidth underutilization, and TTFT budgets)
   automatically. Detection uses stock llama-bench flags: if op-offload
   on/off makes no difference for a quantized model while full-GPU is much
   faster, the weights are repack-locked.

```
$ python3 edge_audit.py --bench llama-bench --model LFM2-1.2B-Q4_0.gguf
[FAIL] C1 repack-blocks-offload — GPU prefill offload is not engaging
    evidence: offload on = 624 t/s vs off = 624 t/s (gain 1.00x,
              healthy >1.5x); full-GPU = 1968 t/s
    action:   use --no-repack / -DGGML_CPU_REPACK=OFF; expect ~2-2.7x
              faster TTFT at the cost of ~20-25% decode
```

After applying the fix, the same audit reports `[ OK ] ... gain 9.21x`.

Longer-term direction (proposed upstream): make the repack decision
offload-aware, or support dual layouts so GPU prefill and repacked-CPU
decode can coexist.

## Repo layout

```
edge_audit.py            # the auditor (run with --mock for a demo)
bench/                   # Phase 1 harness: setup.sh, run_bench.sh, analyze.py
profiling/               # Phase 3: profile_prefill.sh + PHASE3.md guide
results/                 # raw llama-bench JSON + audit reports (before/after)
docs/                    # investigation notes, github-issue-draft.md
```

## Reproduce

```bash
# baseline harness
./bench/setup.sh && ./bench/run_bench.sh
python3 bench/analyze.py results/<timestamp>

# the A/B (requires a second build with -DGGML_CPU_REPACK=OFF)
llama-bench -m models/LFM2-1.2B-Q4_0.gguf -p 512 -n 128 -r 10 -t 5 -ngl 0

# the auditor
python3 edge_audit.py --bench <llama-bench> --model <model.gguf>
```

Methodology: idle machine, AC power, 10 repetitions, medians reported,
configs with CV > 5% flagged and rerun. Environment recorded per run.

---

*Measured on Apple M3 Pro (5P+6E, 18 GB, macOS 26.5.1), llama.cpp b9993
(2969d6d15), July 2026. LFM2 weights by Liquid AI (LFM Open License).*
