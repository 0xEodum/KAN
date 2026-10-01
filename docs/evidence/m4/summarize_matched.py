"""Pool the frozen balanced full-call samples; preserve every input row."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
runs = [("preopt", 1), ("final", 1), ("final", 2), ("preopt", 2)]
rows = []
for version, run in runs:
    with (ROOT / f"balanced-{version}-{run}.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["workspace_allocations"]) != 2 or float(row["max_abs_error"]) > 2e-10:
                raise RuntimeError("Verification/allocation invariant failed")
            rows.append({"version": version, "run": run, **row})
with (ROOT / "balanced-final.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def quantile(values, fraction):
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

summary = {"order": runs, "warmups": 2, "repeats_per_run": 7, "modes": {}}
for mode in ("m4_resident_full", "m4_transfer_full"):
    summary["modes"][mode] = {}
    for version in ("preopt", "final"):
        selected = [r for r in rows if r["version"] == version and r["backend"] == mode]
        samples = [float(x) for r in selected for x in r["samples_ms"].split(";")]
        summary["modes"][mode][version] = {
            "samples": samples,
            "median_ms": quantile(samples, 0.5),
            "iqr_ms": quantile(samples, 0.75) - quantile(samples, 0.25),
            "max_abs_error": max(float(r["max_abs_error"]) for r in selected),
        }
    baseline = summary["modes"][mode]["preopt"]["median_ms"]
    final = summary["modes"][mode]["final"]["median_ms"]
    summary["modes"][mode]["median_reduction_percent"] = 100 * (baseline - final) / baseline
(ROOT / "balanced-final-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
