"""Per-launch-shape medians of the wavelet forward/backward kernels from an nsys
SQLite export (nsys export -t sqlite): grid (x, y, z) identifies the layer."""
import collections, sqlite3, statistics, sys
c = sqlite3.connect(sys.argv[1])
d = collections.defaultdict(list)
for name, gx, gy, gz, dur in c.execute("select s.value, k.gridX, k.gridY, k.gridZ, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName = s.id"):
    if 'wavelet_forward' in name or 'wavelet_backward' in name:
        d[(name.split('(')[0].replace('void ', '').replace('kan::cuda::<unnamed>::', ''), gx, gy, gz)].append(dur)
print('kernel,grid,launches,median_us')
for k, v in sorted(d.items()):
    print(f'"{k[0]}",{k[1]}x{k[2]}x{k[3]},{len(v)},{statistics.median(v) / 1000:.1f}')
