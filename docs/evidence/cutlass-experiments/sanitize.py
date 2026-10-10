"""Bounded additional sanitizer checks, sequentially after timing/profiling."""
import json
import shutil
import subprocess
import time
from run import HERE, EXE, ensure_environment


def main():
    ensure_environment()
    results = []
    for tool in ("racecheck", "synccheck", "initcheck"):
        command = [shutil.which("compute-sanitizer"), "--tool", tool,
                   "--error-exitcode", "99", str(EXE), "check", "4", "2",
                   "f32", "resident", "small", "1", "0"]
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = process.communicate(timeout=240)
            verdict = "passed" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            output, _ = process.communicate()
            verdict = "timeout-no-result"
        (HERE/f"sanitizer-{tool}.log").write_text(output, encoding="utf8")
        results.append(dict(tool=tool, verdict=verdict, exit=process.returncode,
                            seconds=time.monotonic()-start, timeout_seconds=240, command=command))
        print(tool, verdict, flush=True)
        (HERE/"sanitizer-results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf8")
    if any(r["verdict"] != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
