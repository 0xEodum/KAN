"""Compare per-kernel SASS of two resident.cu objects (backlog C5): every kernel
present in both must be instruction-identical (addresses stripped; addresses of
any width: up to the M7 review only the first 64 KB of a kernel, four-digit
addresses, were compared).
Usage: python sass_cmp.py <before.o> <after.o>"""
import re, subprocess, sys
def kernels(obj):
    text = subprocess.run(['cuobjdump', '-sass', obj], capture_output=True, text=True).stdout
    out, name, body = {}, None, []
    for line in text.splitlines():
        m = re.match(r'\s+Function : (\S+)', line)
        if m:
            if name: out[name] = body
            name, body = re.sub(r'_GLOBAL__N__[0-9a-f]+_11_resident_cu_[0-9a-f]{8}', '_GLOBAL__N__resident', m.group(1)), []
        elif name and re.search(r'/\*[0-9a-f]{4,}\*/', line):
            # Constant bank 4 offsets are relocations of global tables (they move
            # when the TU gains globals, e.g. CUB's); instructions are compared.
            body.append(re.sub(r'c\[0x4\]\[[^\]]*\]', 'c[0x4][*]', re.sub(r'/\*[0-9a-f]{4,}\*/|;\s*/\*.*', '', line)).strip())
    if name: out[name] = body
    return out
a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
common = sorted(a.keys() & b.keys())
diff = [k for k in common if a[k] != b[k]]
print(f'before {len(a)} kernels, after {len(b)}, common {len(common)}, differing {len(diff)}')
for k in diff: print('  differs:', k[:120])
print('only after:', len(b.keys() - a.keys()), 'e.g.', sorted(b.keys() - a.keys())[:3])
print('only before:', sorted(a.keys() - b.keys()))
