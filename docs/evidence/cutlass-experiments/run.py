"""Sequential GPU experiments; durable per-process ledger and resumable runs.

python docs/evidence/cutlass-experiments/run.py screen|confirm|learn|summarize
Large profiler files live under build-cutlass-experiment, not this directory.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXE = ROOT / "build-cutlass-experiment/cutlass_experiment.exe"
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)
CASES = ["tiny", "small", "irregular", "deep", "medium", "wide", "large"]
STEPS = dict(tiny=400, small=300, irregular=400, deep=100, medium=80, wide=40, large=10)
MODE_NAMES = {0: "cuBLAS", 1: "CUTLASS-dense-forward", 2: "CUTLASS-input-fusion",
              3: "CUTLASS-virtual-forward", 4: "CUTLASS-full-fusion", 5: "CUTLASS-residual-fusion"}


def run(args, label, timeout=600):
    path = RAW / (label + ".txt")
    if path.exists() and path.read_text(encoding="utf8").endswith("\nEXIT=0\n"):
        return path
    start = time.time()
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    path.write_text(result.stdout + result.stderr + f"\nEXIT={result.returncode}\n", encoding="utf8")
    with (HERE / "ledger.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(dict(label=label, command=[str(EXE), *args], exit=result.returncode,
                               seconds=time.time()-start, sha256=hashlib.sha256(path.read_bytes()).hexdigest())) + "\n")
    print(label, "exit", result.returncode, "wall", round(time.time()-start, 2), flush=True)
    return path


def ok(path):
    return path.read_text(encoding="utf8").endswith("\nEXIT=0\n")


def rows(path):
    lines = [x for x in path.read_text(encoding="utf8").splitlines() if x and not x.startswith("EXIT=")]
    return list(csv.DictReader(lines)) if lines and lines[0].startswith("mode,") else []


def median(path):
    return statistics.median(float(r["ms_per_step"]) for r in rows(path))


def screen():
    # Every configuration is checked before its performance enters selection.
    selections = {}
    for mode in range(1, 6):
        scores = []
        for tile in range(3):
            paths = [run(["check", mode, tile, p, "resident", c, b, 0],
                         f"check-m{mode}-t{tile}-{p}-{c}-b{b}")
                     for p in ("f32", "tf32") for c in ("small", "tail", "wide") for b in (0, 1)]
            if not all(ok(p) for p in paths):
                scores.append(dict(tile=tile, correct=False))
                continue
            timings = []
            for c in ("medium", "wide", "large"):
                a = run(["bench", 0, tile, "f32", "resident", c, 1, 0, 3, STEPS[c]], f"screen-m{mode}-t{tile}-{c}-a")
                b = run(["bench", mode, tile, "f32", "resident", c, 1, 0, 3, STEPS[c]], f"screen-m{mode}-t{tile}-{c}-b")
                if not ok(a) or not ok(b):
                    break
                timings.append(median(b)/median(a))
            if len(timings) == 3:
                scores.append(dict(tile=tile, correct=True, ratios=timings,
                                   score=statistics.geometric_mean(timings)))
        valid = [s for s in scores if s.get("correct") and "score" in s]
        selections[str(mode)] = dict(scores=scores, tile=min(valid, key=lambda s: s["score"])["tile"] if valid else None)
        (HERE / "selection.json").write_text(json.dumps(selections, indent=2)+"\n", encoding="utf8")
    return selections


def confirm():
    selected = json.loads((HERE / "selection.json").read_text())
    # Paired ABBA within each independent initialization; same five windows
    # and update count. Separate p/c/b/protocol groups prevent mixed claims.
    for c in CASES:
        for precision in ("f32", "tf32"):
            for protocol in ("resident", "host"):
                for branch in (0, 1):
                    for seed in range(3):
                        for mode in range(1, 6):
                            tile = selected[str(mode)]["tile"]
                            if tile is None:
                                continue
                            if mode == 5 and branch == 0:
                                continue  # no residual work exists
                            for order, variant in enumerate((0, mode, mode, 0)):
                                label = f"confirm-{c}-{precision}-{protocol}-b{branch}-s{seed}-m{mode}-o{order}"
                                path = run(["bench", variant, tile, precision, protocol, c, branch, seed, 5, STEPS[c]], label)
                                if not ok(path):
                                    raise RuntimeError("confirmation failed: " + label)


def learn():
    selected = json.loads((HERE / "selection.json").read_text())
    for c in ("small", "deep", "medium"):
        for p in ("f32", "tf32"):
            for branch in (0, 1):
                for seed in range(3):
                    for mode in range(6):
                        tile = 0 if mode == 0 else selected[str(mode)]["tile"]
                        if tile is None or (mode == 5 and branch == 0):
                            continue
                        path = run(["train", mode, tile, p, "resident", c, branch, seed, 1, 1000],
                                   f"learn-{c}-{p}-b{branch}-s{seed}-m{mode}")
                        if not ok(path):
                            raise RuntimeError("learning failed: " + str(path))


def summarize():
    samples = []
    for path in sorted(RAW.glob("confirm-*.txt")):
        if ok(path):
            group = path.stem.split("-")
            comparison_mode, order = int(group[-2][1:]), int(group[-1][1:])
            for r in rows(path):
                samples.append(dict(**r, comparison_mode=comparison_mode, order=order, file=path.name))
    if not samples:
        return
    with (HERE / "samples.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    groups = {}
    for r in samples:
        key = (r["case"], r["precision"], r["protocol"], r["branch"], r["comparison_mode"])
        groups.setdefault(key, []).append(r)
    summary = []
    for key, rs in groups.items():
        control = [float(r["ms_per_step"]) for r in rs if int(r["mode"]) == 0]
        candidate = [float(r["ms_per_step"]) for r in rs if int(r["mode"]) != 0]
        ratios = []
        for seed in range(3):
            a = [float(r["ms_per_step"]) for r in rs if int(r["seed"]) == seed and int(r["mode"]) == 0]
            b = [float(r["ms_per_step"]) for r in rs if int(r["seed"]) == seed and int(r["mode"]) != 0]
            if a and b:
                ratios.append(statistics.median(b)/statistics.median(a))
        if control and candidate:
            summary.append(dict(case=key[0], precision=key[1], protocol=key[2], branch=key[3], mode=key[4],
                                control_ms=statistics.median(control), candidate_ms=statistics.median(candidate),
                                ratio=statistics.median(ratios), seed_ratio_min=min(ratios), seed_ratio_max=max(ratios),
                                control_min=min(control), control_max=max(control), candidate_min=min(candidate),
                                candidate_max=max(candidate), samples=len(rs)))
    with (HERE / "summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# Complete training-step confirmation", "", "Ratio = CUTLASS / control wall time; lower is better.", "",
             "| Case | Precision | Protocol | Branch | Mode | Control ms | CUTLASS ms | Ratio | Seed range |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        lines.append(f"| {r['case']} | {r['precision']} | {r['protocol']} | {r['branch']} | {r['mode']} | "
                     f"{r['control_ms']:.4f} | {r['candidate_ms']:.4f} | {r['ratio']:.3f} | "
                     f"{r['seed_ratio_min']:.3f}–{r['seed_ratio_max']:.3f} |")
    (HERE / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    {"screen": screen, "confirm": confirm, "learn": learn, "summarize": summarize}[sys.argv[1]]()
