#!/usr/bin/env python3


import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

TTFT_BUDGET_MS = 100.0
OFFLOAD_GAIN_HEALTHY = 1.5   # offload-on/off prefill ratio below this = suspicious
OVERSUB_TOLERANCE = 0.95     # all-cores must beat P-cores by >5% to be worth it


def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


@dataclass
class Hardware:
    chip: str = "unknown"
    p_cores: int = 0
    e_cores: int = 0
    total_cores: int = 0
    mem_gb: float = 0.0
    peak_bw_gbps: float = 0.0   # 0 = unknown
    os: str = ""

    @staticmethod
    def probe():
        hw = Hardware(os=f"{platform.system()} {platform.release()}")
        if platform.system() == "Darwin":
            hw.chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
            hw.p_cores = int(_sysctl("hw.perflevel0.physicalcpu") or 0)
            hw.e_cores = int(_sysctl("hw.perflevel1.physicalcpu") or 0)
            hw.total_cores = int(_sysctl("hw.ncpu") or 0)
            hw.mem_gb = int(_sysctl("hw.memsize") or 0) / 2**30
            hw.peak_bw_gbps = KNOWN_BW.get(_short_chip(hw.chip), 0.0)
        else:
            hw.chip = platform.processor() or platform.machine()
            hw.total_cores = __import__("os").cpu_count() or 0
            hw.p_cores = hw.total_cores
        return hw


def _short_chip(brand):
    matches = [k for k in KNOWN_BW if k.lower() in (brand or "").lower()]
    return max(matches, key=len) if matches else brand


# Published peak memory bandwidth (GB/s) for common Apple chips.
# Used only for the C3 roofline estimate; unknown chips skip that check.
KNOWN_BW = {
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
}


@dataclass
class BenchResult:
    label: str
    pp_tps: float = 0.0     # prefill tokens/sec (pp512)
    tg_tps: float = 0.0     # decode tokens/sec (tg128)
    pp_stddev: float = 0.0
    tg_stddev: float = 0.0
    model_size: int = 0


class BenchRunner:
    def __init__(self, bench_path, model_path, reps=5):
        self.bench = bench_path
        self.model = model_path
        self.reps = reps

    def run(self, label, extra_args):
        cmd = [self.bench, "-m", self.model, "-p", "512", "-n", "128",
               "-r", str(self.reps), "-o", "json"] + extra_args
        print(f"  running {label}: {' '.join(extra_args)}", file=sys.stderr)
        out = subprocess.check_output(cmd, text=True,
                                      stderr=subprocess.DEVNULL, timeout=1800)
        return self._parse(label, json.loads(out))

    @staticmethod
    def _parse(label, tests):
        r = BenchResult(label=label)
        for t in tests:
            samples = t.get("samples_ts") or []
            avg = statistics.median(samples) if samples else t.get("avg_ts", 0.0)
            sd = (statistics.stdev(samples) if len(samples) > 1
                  else t.get("stddev_ts", 0.0))
            if t.get("n_prompt") and not t.get("n_gen"):
                r.pp_tps, r.pp_stddev = avg, sd
            elif t.get("n_gen") and not t.get("n_prompt"):
                r.tg_tps, r.tg_stddev = avg, sd
            r.model_size = t.get("model_size", r.model_size)
        return r


class MockRunner:
    """Synthetic results mirroring real M3 Pro measurements (LFM2-1.2B-Q4_0),
    so the report pipeline can be demoed and tested without hardware."""
    DATA = {
        "gpu_full":        (1966.0, 163.0),
        "cpu_offload_on":  (620.0, 155.0),
        "cpu_offload_off": (612.0, 154.0),
        "cpu_t_pcores":    (617.0, 152.0),
        "cpu_t_all":       (541.0, 110.0),
    }

    def __init__(self):
        self.model = "mock/LFM2-1.2B-Q4_0.gguf"

    def run(self, label, extra_args):
        pp, tg = self.DATA.get(label, (500.0, 100.0))
        return BenchResult(label=label, pp_tps=pp, tg_tps=tg,
                           pp_stddev=pp * 0.02, tg_stddev=tg * 0.02,
                           model_size=693_000_000)


@dataclass
class Finding:
    check: str
    severity: str           # "pathology" | "warning" | "info" | "ok"
    title: str
    evidence: str
    recommendation: str


def check_repack_offload(res, hw, model_name):
    """C1: at -ngl 0 with a GPU present, op-offload should accelerate prefill.
    If offload-on ~= offload-off for a quantized model, weights are likely in
    a repacked CPU-only layout the GPU cannot read."""
    on, off = res.get("cpu_offload_on"), res.get("cpu_offload_off")
    gpu = res.get("gpu_full")
    if not (on and off):
        return None
    if not gpu or gpu.pp_tps <= 0:
        return None  
    gain = on.pp_tps / off.pp_tps if off.pp_tps else 0
    quantized = any(q in model_name.upper() for q in
                    ("Q4", "Q5", "Q8", "Q2", "Q3", "Q6", "IQ"))
    if gain < OFFLOAD_GAIN_HEALTHY and quantized and gpu.pp_tps > 2 * on.pp_tps:
        return Finding(
            check="C1 repack-blocks-offload",
            severity="pathology",
            title="GPU prefill offload is not engaging for this quantized model",
            evidence=(f"prefill with op-offload on = {on.pp_tps:.0f} t/s vs off = "
                      f"{off.pp_tps:.0f} t/s (gain {gain:.2f}x, healthy is >"
                      f"{OFFLOAD_GAIN_HEALTHY}x); full-GPU = {gpu.pp_tps:.0f} t/s"),
            recommendation=(
                "Weights are likely repacked into a CPU-only layout at load "
                "time, which blocks batched-matmul GPU offload. If running "
                "CPU-mostly on purpose: use --no-repack (llama-cli/server) or "
                "build with -DGGML_CPU_REPACK=OFF. Expect ~2-2.7x faster "
                "prefill/TTFT at the cost of ~20-25% decode. Prefer full GPU "
                "offload (-ngl 99) when the GPU is free."))
    return Finding(check="C1 repack-blocks-offload", severity="ok",
                   title="GPU op-offload behaves normally at -ngl 0",
                   evidence=f"offload on/off prefill gain = {gain:.2f}x",
                   recommendation="No action.")


def check_thread_oversubscription(res, hw):
    p, a = res.get("cpu_t_pcores"), res.get("cpu_t_all")
    if not (p and a) or hw.e_cores == 0:
        return None
    decode_ratio = a.tg_tps / p.tg_tps if p.tg_tps else 1.0
    noisy = a.tg_stddev > 2 * p.tg_stddev
    if decode_ratio < OVERSUB_TOLERANCE:
        return Finding(
            check="C2 thread oversubscription",
            severity="warning",
            title=f"Using all {hw.total_cores} cores is slower than "
                  f"{hw.p_cores} P-cores",
            evidence=(f"decode: {a.tg_tps:.0f} t/s at t={hw.total_cores} vs "
                      f"{p.tg_tps:.0f} t/s at t={hw.p_cores} "
                      f"({(1-decode_ratio)*100:.0f}% slower"
                      + (", higher variance" if noisy else "") + ")"),
            recommendation=(f"Set threads to the P-core count (-t {hw.p_cores}). "
                            "Barrier-synchronized kernels run at the pace of the "
                            "slowest core; E-cores create stragglers."))
    return Finding(check="C2 thread oversubscription", severity="ok",
                   title="Thread count scaling looks healthy",
                   evidence=f"all-cores/P-cores decode ratio = {decode_ratio:.2f}",
                   recommendation="No action.")


def check_bandwidth(res, hw):
    best = max((r for r in res.values() if r.tg_tps and r.model_size),
               key=lambda r: r.tg_tps, default=None)
    if not best or not hw.peak_bw_gbps:
        return None
    implied = best.model_size * best.tg_tps / 1e9
    util = implied / hw.peak_bw_gbps
    sev = "info" if util > 0.6 else "warning"
    title = ("Decode is near the memory-bandwidth roofline"
             if util > 0.6 else "Decode is far from the bandwidth roofline")
    rec = ("Expected for well-optimized decode; further decode gains require "
           "a smaller quant, not faster kernels."
           if util > 0.6 else
           "Decode has headroom: check thread config, backend choice, and "
           "whether another process is contending for memory bandwidth.")
    return Finding(check="C3 bandwidth utilization", severity=sev, title=title,
                   evidence=(f"best decode {best.tg_tps:.0f} t/s x "
                             f"{best.model_size/1e6:.0f} MB = {implied:.0f} GB/s "
                             f"= {util*100:.0f}% of ~{hw.peak_bw_gbps:.0f} GB/s peak"),
                   recommendation=rec)


def check_ttft_budget(res):
    lines = []
    for r in sorted(res.values(), key=lambda r: -r.pp_tps):
        if r.pp_tps:
            lines.append(f"{r.label}: {int(r.pp_tps * TTFT_BUDGET_MS / 1000)} tokens")
    if not lines:
        return None
    return Finding(check="C4 TTFT budget", severity="info",
                   title=f"Max prompt length within {TTFT_BUDGET_MS:.0f} ms TTFT",
                   evidence="; ".join(lines),
                   recommendation=("Pick the config whose budget covers your "
                                   "typical prompt length."))



def build_matrix(runner, hw):
    t_p = str(hw.p_cores or 4)
    t_all = str(hw.total_cores or 8)
    matrix = {
        "gpu_full":        ["-ngl", "99"],
        "cpu_offload_on":  ["-ngl", "0", "-t", t_p],
        "cpu_offload_off": ["-ngl", "0", "-t", t_p, "-nopo", "1"],
        "cpu_t_all":       ["-ngl", "0", "-t", t_all, "-nopo", "1"],
    }
    results = {}
    for label, args in matrix.items():
        try:
            results[label] = runner.run(label, args)
        except Exception as e:
            print(f"  ! {label} failed: {e}", file=sys.stderr)
    if "cpu_offload_off" in results:
        r = results["cpu_offload_off"]
        results["cpu_t_pcores"] = BenchResult(
            label="cpu_t_pcores", pp_tps=r.pp_tps, tg_tps=r.tg_tps,
            pp_stddev=r.pp_stddev, tg_stddev=r.tg_stddev,
            model_size=r.model_size)
    return results


def render_report(hw, model_name, results, findings):
    icon = {"pathology": "[FAIL]", "warning": "[WARN]",
            "info": "[INFO]", "ok": "[ OK ]"}
    out = ["# edge-audit report", "",
           f"- device: {hw.chip} ({hw.p_cores}P+{hw.e_cores}E, "
           f"{hw.mem_gb:.0f} GB, ~{hw.peak_bw_gbps or '?'} GB/s)",
           f"- model: {model_name}", "",
           "## Measurements (pp512 / tg128, tokens/sec)", ""]
    for r in results.values():
        out.append(f"- {r.label}: prefill {r.pp_tps:.0f} +/- {r.pp_stddev:.0f}, "
                   f"decode {r.tg_tps:.0f} +/- {r.tg_stddev:.0f}")
    out += ["", "## Findings", ""]
    order = {"pathology": 0, "warning": 1, "info": 2, "ok": 3}
    for f in sorted(findings, key=lambda f: order[f.severity]):
        out += [f"{icon[f.severity]} {f.check} — {f.title}",
                f"    evidence: {f.evidence}",
                f"    action:   {f.recommendation}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="edge-audit v0")
    ap.add_argument("--bench", help="path to llama-bench binary")
    ap.add_argument("--model", help="path to a GGUF model")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mock", action="store_true",
                    help="demo mode with synthetic M3 Pro data")
    ap.add_argument("--json", help="also write findings to this JSON path")
    args = ap.parse_args()

    hw = Hardware.probe()
    if args.mock:
        runner = MockRunner()
        hw = Hardware(chip="Apple M3 Pro (mock)", p_cores=5, e_cores=6,
                      total_cores=11, mem_gb=18, peak_bw_gbps=150,
                      os="Darwin (mock)")
    else:
        if not (args.bench and args.model):
            ap.error("--bench and --model are required (or use --mock)")
        if not shutil.which(args.bench) and not Path(args.bench).exists():
            ap.error(f"llama-bench not found at {args.bench}")
        runner = BenchRunner(args.bench, args.model, args.reps)

    print("edge-audit: running benchmark matrix "
          "(keep the machine idle, on AC power)...", file=sys.stderr)
    results = build_matrix(runner, hw)

    model_name = Path(runner.model).name
    findings = [f for f in (
        check_repack_offload(results, hw, model_name),
        check_thread_oversubscription(results, hw),
        check_bandwidth(results, hw),
        check_ttft_budget(results),
    ) if f]

    report = render_report(hw, model_name, results, findings)
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"hardware": asdict(hw),
             "results": {k: asdict(v) for k, v in results.items()},
             "findings": [asdict(f) for f in findings]}, indent=2))
        print(f"(json written to {args.json})", file=sys.stderr)


if __name__ == "__main__":
    main()
