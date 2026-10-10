"""Audit the archived experiment without a GPU or downloaded CUTLASS tree."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
HERE = Path(os.environ.get("KAN_EXPERIMENT_EVIDENCE", str(SOURCE))).resolve()
ROOT = SOURCE.parents[2]


def digest(path):
    data = path.read_bytes()
    if b"\0" not in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read(name):
    with (HERE/name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((HERE/"manifest.json").read_text())
    for field, root in (("source_sha256", ROOT), ("evidence_sha256", HERE)):
        for name, expected in manifest[field].items():
            assert digest(root/name) == expected, f"digest mismatch: {name}"
    expected_counts = {"confirm": 3024, "adaptive-confirm": 384,
                       "adaptive-screen": 64, "learn": 198}
    for prefix, expected in expected_counts.items():
        paths = list((HERE/"raw").glob(prefix+"-*.txt"))
        assert len(paths) == expected, (prefix, len(paths), expected)
        assert all(p.read_text().endswith("\nEXIT=0\n") for p in paths), prefix
    for filename, expected in (("samples.csv", 15120), ("adaptive-samples.csv", 1920)):
        rows = read(filename)
        assert len(rows) == expected, filename
        assert all(math.isfinite(float(r["ms_per_step"])) and float(r["ms_per_step"]) > 0 for r in rows)
    for filename, expected in (("summary.csv", 252), ("adaptive-summary.csv", 32)):
        rows = read(filename)
        assert len(rows) == expected, filename
        assert all(int(r["samples"]) == 60 for r in rows), filename
    assert all(r["verdict"] == "PASS" for r in read("correctness-summary.csv"))
    learning = read("learning-summary.csv")
    assert len(learning) == 162
    assert all(math.isfinite(float(r["candidate_loss"])) and int(r["parameter_values"]) > 0 for r in learning)
    print("PASS: hashes, complete paired matrices, correctness and learning evidence")


if __name__ == "__main__":
    main()
