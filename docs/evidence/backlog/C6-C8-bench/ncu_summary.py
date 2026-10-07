"""One line per profiled kernel from an ncu_profile.sh CSV (long format)."""
import csv
import re
import sys
from collections import OrderedDict


def short(name):
    m = re.search(r"(rational_\w+_kernel)", name)
    return m.group(1) if m else name[:40]


def main(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    kernels = OrderedDict()
    for row in csv.DictReader(lines[start:]):
        k = kernels.setdefault(row["ID"], {"name": short(row["Kernel Name"]), "grid": row["Grid Size"]})
        k[row["Metric Name"]] = row["Metric Value"].replace(",", "")
    f = lambda k, m: float(k.get(m, "nan"))
    print("kernel                      grid            us     SM%  DRAM%  DRAM MB(r/w)   ld sect/req st sect/req  L1hit%  L2hit%  occ%  regs  fp64%")
    for k in kernels.values():
        ld = f(k, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / max(1, f(k, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))
        st = f(k, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") / max(1, f(k, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"))
        print(f"{k['name']:27s} {k['grid']:14s} {f(k,'gpu__time_duration.sum')/1e3:9.1f} "
              f"{f(k,'sm__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f} {f(k,'dram__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f} "
              f"{f(k,'dram__bytes_read.sum')/1e6:7.1f}/{f(k,'dram__bytes_write.sum')/1e6:<7.1f} {ld:9.2f} {st:11.2f} "
              f"{f(k,'l1tex__t_sector_hit_rate.pct'):7.1f} {f(k,'lts__t_sector_hit_rate.pct'):7.1f} "
              f"{f(k,'sm__warps_active.avg.pct_of_peak_sustained_active'):5.1f} {f(k,'launch__registers_per_thread'):5.0f} "
              f"{f(k,'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active'):6.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
