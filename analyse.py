import csv
import json
import statistics as stats
import sys
from pathlib import Path


def load_runs(result_dir: Path):
    rows = []
    for f in sorted(result_dir.glob("*.json")):
        label = f.stem  #
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! skipping unparseable {f.name}", file=sys.stderr)
            continue
        if isinstance(data, dict):
            data = [data]

        model, _, backend = label.partition("__")
        for test in data:
            n_prompt = test.get("n_prompt", 0)
            n_gen = test.get("n_gen", 0)
            if n_prompt and n_gen:
                phase = "pp+tg"     
            elif n_prompt:
                phase = "prefill"
            elif n_gen:
                phase = "decode"
            else:
                continue

            samples = test.get("samples_ts") or []
            if samples:
                avg = stats.median(samples)         
                sd = stats.stdev(samples) if len(samples) > 1 else 0.0
            else:
                avg = test.get("avg_ts", 0.0)
                sd = test.get("stddev_ts", 0.0)
            if not avg:
                continue

            row = {
                "model": model,
                "backend": backend,
                "phase": phase,
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                "tokens_per_sec": round(avg, 2),
                "stddev": round(sd, 2),
                "cv_pct": round(100.0 * sd / avg, 1) if avg else 0.0,
                "n_samples": len(samples) or test.get("reps", ""),
                "model_size_bytes": test.get("model_size", ""),
            }
            if phase == "prefill":
                row["ttft_ms"] = round(n_prompt / avg * 1000.0, 1)
            rows.append(row)
    return rows


def fmt_table(rows, headers):
    if not rows:
        return "_no data_\n"
    widths = [max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")).ljust(w)
                                     for h, w in zip(headers, widths)) + " |")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result_dir = Path(sys.argv[1])
    rows = load_runs(result_dir)
    if not rows:
        sys.exit(f"No usable llama-bench JSON found in {result_dir}")

    rows.sort(key=lambda r: (r["model"], r["backend"], r["phase"], r["n_prompt"]))
    prefill = [r for r in rows if r["phase"] == "prefill"]
    decode = [r for r in rows if r["phase"] == "decode"]
    noisy = [r for r in rows if r["cv_pct"] > 5.0]

    env = (result_dir / "env.txt")
    env_txt = env.read_text() if env.exists() else "(env.txt missing)"

    report = ["# LFM2 on Apple M3 Pro — Phase 1 baseline\n",
              "## Environment\n```\n" + env_txt + "```\n",
              "## Prefill (prompt processing) & derived TTFT\n",
              fmt_table(prefill, ["model", "backend", "n_prompt",
                                  "tokens_per_sec", "ttft_ms", "cv_pct"]),
              "\n## Decode (token generation)\n",
              fmt_table(decode, ["model", "backend", "n_gen",
                                 "tokens_per_sec", "cv_pct"])]

    if noisy:
        report.append("\n## ⚠ Noisy configs (CV > 5% — rerun before publishing)\n")
        report.append(fmt_table(noisy, ["model", "backend", "phase",
                                        "n_prompt", "n_gen", "cv_pct"]))

    bw_rows = []
    for r in decode:
        size = r.get("model_size_bytes")
        if size:
            gbps = float(size) * r["tokens_per_sec"] / 1e9
            bw_rows.append({"model": r["model"], "backend": r["backend"],
                            "tokens_per_sec": r["tokens_per_sec"],
                            "implied_GB_per_s": round(gbps, 1),
                            "pct_of_150GBps_roofline": round(gbps / 150.0 * 100, 1)})
    if bw_rows:
        report.append("\n## Decode bandwidth sanity check "
                      "(M3 Pro peak ~150 GB/s)\n")
        report.append(fmt_table(bw_rows, ["model", "backend", "tokens_per_sec",
                                          "implied_GB_per_s",
                                          "pct_of_150GBps_roofline"]))
        report.append("\n_Implied BW = model_size x decode t/s. If this is far "
                      "below the roofline, decode has headroom; if close, "
                      "it's bandwidth-saturated (expected)._\n")

    (result_dir / "report.md").write_text("\n".join(report))

    all_keys = ["model", "backend", "phase", "n_prompt", "n_gen",
                "tokens_per_sec", "stddev", "cv_pct", "ttft_ms",
                "n_samples", "model_size_bytes"]
    with open(result_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})

    print(f"Wrote {result_dir/'report.md'} and {result_dir/'results.csv'}")
    print(f"{len(rows)} measurements, {len(noisy)} flagged as noisy (CV>5%)")


if __name__ == "__main__":
    main()
