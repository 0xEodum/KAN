"""Record bytes, source provenance and local build artifacts after verification."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/evidence/m4"

def record(path):
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}

def command(*arguments):
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()

def git_record(revision, path):
    data = subprocess.check_output(("git", "show", f"{revision}:{path}"), cwd=ROOT)
    return {"revision": revision, "path": path, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob": command("git", "rev-parse", f"{revision}:{path}")}

manifest = {
    "stage": "M4", "date": "2026-10-01",
    "source": {
        "integration_head": command("git", "rev-parse", "HEAD"),
        "resident_baseline": "c5a54e3", "resident_tuned": "0071628",
        "cpu_for_balanced_gpu": "4b808d2", "frozen_fixture": "73e3f47",
        "balanced_source_blobs": [git_record(revision, path) for revision, path in (
            ("c5a54e3", "src/resident.cu"), ("0071628", "src/resident.cu"),
            ("4b808d2", "src/rational.cpp"), ("4b808d2", "src/layer.cpp"),
            ("73e3f47", "benchmarks/m4_benchmark.cpp"))],
        "files": [record(ROOT / path) for path in (
            "src/resident.cu", "src/rational.cpp", "src/layer.cpp",
            "src/rational_internal.hpp",
            "include/kan/rational.hpp", "include/kan/layer.hpp",
            "benchmarks/m4_benchmark.cpp", "tests/m4_resident_test.cpp")],
    },
    "environment": {
        "gpu": "NVIDIA RTX3090,24576MiB,driver591.86,WindowsWDDM",
        "cpu": "Intel i5-12400,6cores,12logical processors",
        "toolchain": "MSVC19.50.35724.0,CUDA13.1.115,Release,architecture86,--fmad=false,explicit host-compiler override",
        "nsight_systems": "2025.5.2", "nsight_compute": "2025.4.1",
        "counter_limit": "ERR_NVGPUCTRPERM; occupancy/bandwidth counters unavailable",
    },
    "verification": {
        "resident_tests": "7/7", "compute_sanitizer_memcheck_errors": 0,
        "balanced_gpu_order": ["preopt", "final", "final", "preopt"],
        "warmups": 2, "measured_samples_per_run": 7,
        "precision": "binary64", "fixture_change": False,
        "benchmark_checks": "Final pre-update output/allVJPs and final learned parameters; exact untimed evolvingGPUreplay",
    },
    "artifact_notes": {
        "preopt-initial.csv": "Diagnostic initial single run before rare-derivative fix, CPU4b808d2 precursor; not tuning acceptance",
        "tuned-initial.csv": "Diagnostic single run of warp prototype; not tuning acceptance",
        "balanced-*.csv": "Accepted c5a54e3/0071628 GPU comparison, sameCPU4b808d2, frozen mathematical fixtures",
        "*-profile*": "Diagnostic NsightSystems traces/exports, not accepted unprofiled timing",
        "final-m2-regression.csv": "Numerical/allocations regression only; CPU refinements may run concurrently",
        "final-m3-regression.csv": "Numerical/allocations regression only; CPU refinements may run concurrently",
        "final-m4.csv": "Numerical/allocations final integration sweep after CPU refinement; timing diagnostic only due to CPU timing overlap",
    },
    "local_artifact_roles": {
        "build-m4-cuda/m4-preopt-benchmark.exe": "Preserved exact balanced baseline executable,c5a54e3 GPU/4b808d2 CPU",
        "build-m4-cuda/m4_benchmark.exe": "Final integration executable,relinked with c80b7d3 CPU after balanced GPU measurements; not the original tuned executable hash",
        "build-m4-cuda/kan_cuda.lib": "Unchanged tuned0071628 GPU library from accepted balanced runs",
        "build-m4-cuda/m4-preopt-profile.nsys-rep": "Original baseline diagnostic trace",
        "build-m4-cuda/m4-final-profile.nsys-rep": "Original tuned diagnostic trace before CPU refinement",
    },
    "retained": [record(path) for path in sorted(EVIDENCE.iterdir())
                 if path.is_file() and path.name != "manifest.json"],
    "local_reproducible_not_committed": [record(ROOT / path) for path in (
        "build-m4-cuda/m4-preopt-benchmark.exe", "build-m4-cuda/m4_benchmark.exe",
        "build-m4-cuda/kan_cuda.lib",
        "build-m4-cuda/resident-preopt.cu", "build-m4-cuda/m4-preopt-profile.nsys-rep",
        "build-m4-cuda/m4-final-profile.nsys-rep") if (ROOT / path).exists()],
}
(EVIDENCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {len(manifest['retained'])} retained artifacts")
