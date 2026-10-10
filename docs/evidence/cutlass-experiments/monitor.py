"""Read-only GPU telemetry; stop by creating build-cutlass-experiment/monitor.stop."""
import subprocess
import time
from run import HERE, ROOT, ensure_environment

ensure_environment()
stop = ROOT/"build-cutlass-experiment/monitor.stop"
path = HERE/"gpu-state.csv"
needs_header = not path.exists() or not path.stat().st_size
with path.open("a", encoding="utf8") as out:
    if needs_header:
        out.write("timestamp,utilization.gpu [%],temperature.gpu,memory.used [MiB],power.draw [W],clocks.current.sm [MHz],clocks.current.memory [MHz]\n")
    while not stop.exists():
        row = subprocess.check_output([
            "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,temperature.gpu,memory.used,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader"], text=True)
        out.write(row); out.flush()
        time.sleep(30)
