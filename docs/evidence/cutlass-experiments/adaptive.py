"""Shape-specific input-VJP tile follow-up, without changing the frozen backend.

Screen on seed 6; freeze per-(case,precision) tiles; confirm on new seeds 3-5.
This addresses the global tile selection's small-grid occupancy limitation.
"""
import csv
import json
import statistics
import sys
from run import HERE, RAW, STEPS, run, rows, median, ok

CASES = ("tiny", "small", "irregular", "deep")


def screen():
    selections = []
    for c in CASES:
        for p in ("f32", "tf32"):
            controls = []
            scores = []
            for branch in (0, 1):
                path = run(["bench", 0, 0, p, "resident", c, branch, 6, 5, STEPS[c]],
                           f"adaptive-screen-{c}-{p}-b{branch}-control")
                if not ok(path):
                    raise RuntimeError(str(path))
                controls.append(median(path))
            for tile in range(3):
                ratios = []
                for branch in (0, 1):
                    path = run(["bench", 2, tile, p, "resident", c, branch, 6, 5, STEPS[c]],
                               f"adaptive-screen-{c}-{p}-b{branch}-t{tile}")
                    if not ok(path):
                        raise RuntimeError(str(path))
                    ratios.append(median(path)/controls[branch])
                scores.append(dict(tile=tile, ratios=ratios, score=statistics.geometric_mean(ratios)))
            selections.append(dict(case=c, precision=p, scores=scores,
                                   tile=min(scores, key=lambda r: r["score"])["tile"]))
    (HERE/"adaptive-selection.json").write_text(json.dumps(selections, indent=2)+"\n")


def confirm():
    selected = json.loads((HERE/"adaptive-selection.json").read_text())
    for choice in selected:
        c, p, tile = choice["case"], choice["precision"], choice["tile"]
        for branch in (0, 1):
            for protocol in ("resident", "host"):
                for seed in (3, 4, 5):
                    for order, mode in enumerate((0, 2, 2, 0)):
                        path = run(["bench", mode, tile, p, protocol, c, branch, seed, 5, STEPS[c]],
                                   f"adaptive-confirm-{c}-{p}-{protocol}-b{branch}-s{seed}-o{order}")
                        if not ok(path):
                            raise RuntimeError(str(path))


def summarize():
    groups = {}
    samples = []
    for path in sorted(RAW.glob("adaptive-confirm-*.txt")):
        for r in rows(path):
            r["file"] = path.name
            samples.append(r)
            key = (r["case"], r["precision"], r["protocol"], r["branch"], r["tile"])
            groups.setdefault(key, []).append(r)
    with (HERE/"adaptive-samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    summary = []
    for key, rs in groups.items():
        a = [float(r["ms_per_step"]) for r in rs if r["mode"] == "0"]
        b = [float(r["ms_per_step"]) for r in rs if r["mode"] == "2"]
        ratios = []
        for seed in (3, 4, 5):
            ar = [float(r["ms_per_step"]) for r in rs if r["mode"] == "0" and int(r["seed"]) == seed]
            br = [float(r["ms_per_step"]) for r in rs if r["mode"] == "2" and int(r["seed"]) == seed]
            ratios.append(statistics.median(br)/statistics.median(ar))
        summary.append(dict(case=key[0], precision=key[1], protocol=key[2], branch=key[3], tile=key[4],
                            control_ms=statistics.median(a), candidate_ms=statistics.median(b),
                            ratio=statistics.median(ratios), seed_min=min(ratios), seed_max=max(ratios), samples=len(rs)))
    with (HERE/"adaptive-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)


if __name__ == "__main__":
    {"screen": screen, "confirm": confirm, "summarize": summarize}[sys.argv[1]]()
