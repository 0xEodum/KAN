import ctypes
c = ctypes.WinDLL(r"C:\Windows\System32\nvcuda.dll")
assert c.cuInit(0) == 0
d = ctypes.c_int(); c.cuDeviceGet(ctypes.byref(d), 0)
def a(n): v = ctypes.c_int(); c.cuDeviceGetAttribute(ctypes.byref(v), n, d); return v.value
# CU_DEVICE_ATTRIBUTE_*: MULTIPROCESSOR_COUNT=16, L2_CACHE_SIZE=38, SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO=87, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN=97, MAX_THREADS_PER_MULTIPROCESSOR=39
sm, l2, r, sh, th = a(16), a(38), a(87), a(97), a(39)
print(f"SMs={sm} L2={l2/2**20:.1f}MiB fp64_ratio={r} shared_optin={sh/1024:.0f}KiB threads/SM={th}")
def rule(prec, topo, batch, deg, K):
    w = [int(x) for x in topo.split('x')]; fp64 = prec == 'f64'; tens = prec == 'tf32'; sz = 8 if fp64 else 4
    small = (1 << 23) if fp64 else (1 << 24)  # small_forward_contraction
    bits = 21 if fp64 and r >= 8 else (23 if tens else 22); work = (1 << bits) * sm
    out = []
    for i, o in zip(w, w[1:]):
        ps = i * K * o
        large = batch > small // ps and batch >= (work + ps - 1) // ps
        inl2 = batch * o * sz <= l2 // 2
        ratio = (2.0 if inl2 else (8.0 if r >= 8 else 6.0)) if fp64 else ((5.0 if tens else 4.0) if inl2 else (20.0 if tens else 6.0))
        sel = large and K >= ratio * (deg + 1)
        out.append(f"{i}->{o}:{'S' if sel else 'd'}({'L2' if inl2 else 'dram'} r{ratio:g}{'' if large else ' small'})")
    return ' '.join(out)
for line in open(r"%s/c5-configs.txt" % __import__('os').path.dirname(__file__)):
    p = line.split()
    if p: print(f"{p[0]:5} {p[1]:16} b{p[2]:5} d{p[3]} K{p[4]:>3}: {rule(p[0], p[1], int(p[2]), int(p[3]), int(p[4]))}")
