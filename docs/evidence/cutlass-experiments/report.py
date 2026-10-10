"""Generate exportable figures and hash-verified evidence after all GPU runs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from run import SOURCE, HERE, ROOT, EXE

CASES = ["tiny", "small", "irregular", "deep", "medium", "wide", "large"]
LABELS = ["dense GEMM", "input VJP", "virtual forward", "full fusion", "residual VJP"]


def read(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def figures():
    data = read(HERE/"summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for i, precision in enumerate(("f32", "tf32")):
        for j, protocol in enumerate(("resident", "host")):
            ax = axes[i, j]
            ratios = np.full((len(CASES), 5), np.nan)
            inactive = set()
            for r in data:
                if r["precision"] == precision and r["protocol"] == protocol and r["branch"] == "1":
                    row, col = CASES.index(r["case"]), int(r["mode"])-1
                    fallback = (r["mode"] in ("1", "3") and r["case"] in ("tiny", "irregular")) or (
                        r["mode"] == "5" and r["case"] in ("tiny", "small", "irregular", "deep"))
                    if fallback:
                        inactive.add((row,col))
                    else:
                        ratios[row,col] = float(r["ratio"])
            ax.imshow(ratios, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=.8, vcenter=1, vmax=4))
            ax.set_xticks(range(5), LABELS, rotation=25, ha="right")
            ax.set_yticks(range(len(CASES)), CASES)
            ax.set_title(f"{precision.upper()}, {protocol} MSE, residual on")
            for row in range(len(CASES)):
                for col in range(5):
                    label = "—" if (row,col) in inactive else f"{ratios[row, col]:.2f}"
                    ax.text(col, row, label, ha="center", va="center",
                            color="white" if ratios[row, col] > 2.2 else "black", fontsize=9)
    fig.suptitle("CUTLASS / cuBLAS complete-step wall time\n<1 faster; >1 slower; median of three paired seeds; — unchanged fallback")
    fig.savefig(HERE/"step-ratios.png", dpi=160)
    fig.savefig(HERE/"step-ratios.svg")
    plt.close(fig)


def checks():
    checks = []
    for path in sorted((HERE/"raw").glob("check-*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) == 7 and parts[0] == "check":
                checks.append(dict(file=path.name, quantity=parts[1], values=parts[2],
                                   max_abs=parts[3], max_normalized=parts[4], changed_values=parts[5], verdict=parts[6]))
    with (HERE/"correctness-summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)


def learning_figures():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, case in zip(axes, ("small", "deep", "medium")):
        for mode in range(6):
            losses = []
            for seed in range(3):
                path = HERE/"raw"/f"learn-{case}-f32-b1-s{seed}-m{mode}.txt"
                raw = path.read_text().split("\nEXIT=")[0]
                records = list(csv.DictReader(raw.splitlines()))
                losses.append([float(r["loss"]) for r in records])
            values = np.asarray(losses)
            steps = [int(r["step"]) for r in records]
            label = "cuBLAS" if mode == 0 else LABELS[mode-1]
            ax.plot(steps, values.mean(axis=0), label=label, linewidth=1.5)
            ax.fill_between(steps, values.min(axis=0), values.max(axis=0), alpha=.06)
        ax.set_title(case); ax.set_xlabel("MSE updates"); ax.set_ylabel("Training loss")
        ax.set_yscale("log"); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.suptitle("FP32 resident MSE, residual on; mean and range over three seeds")
    fig.savefig(HERE/"learning-curves.png", dpi=160)
    fig.savefig(HERE/"learning-curves.svg")
    plt.close(fig)


def manifest():
    def sha(p):
        data = p.read_bytes()
        # Match Git blob text normalization, so Windows core.autocrlf does
        # not invalidate archived evidence after a fresh checkout.
        if b"\0" not in data:
            data = data.replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()
    sources = [ROOT/"CMakeLists.txt", ROOT/"src/resident.cu"] + [SOURCE/name for name in (
        "backend.cu", "backend.hpp", "bench.cpp", "build.ps1", "run.py", "adaptive.py",
        "profile.ps1", "report.py", "sanitize.py", "monitor.py", "verify.py")]
    evidence = [p for p in HERE.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    result = dict(digest_format="SHA256: LF-normalized text; exact binary bytes (NUL detected)",
                  control_commit="37a07f8", implementation_commit="b27eecb",
                  cutlass_commit=subprocess.check_output(["git", "-C", str(ROOT/"build-cutlass-deps/cutlass"), "rev-parse", "HEAD"], text=True).strip(),
                  executable_sha256=sha(EXE),
                  source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources},
                  evidence_sha256={str(p.relative_to(HERE)).replace("\\", "/"): sha(p) for p in sorted(evidence)})
    (HERE/"manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    checks()
    figures()
    learning_figures()
    manifest()
