#!/usr/bin/env bash
# Backlog R6: generated-code identity of a build of the base revision and a
# build of the formatted tree with the same options. Host code: every object
# file (.o) of the base build tree against the same path in the new tree,
# function by function (objcmp.py: R4's objdiff.py with string-pool operands
# resolved to their strings; see there for the strings-ws class).
# objcmp.py also compares the constant/data and fatbinary sections byte for byte.
# Device code: every CUDA object, per kernel (C5's sass_cmp.py, full kernels)
# and as whole SASS text (objdiff.py sass). libkan.a / libkan_cuda.a and the
# Python module: functions (objdiff.py) and bytes (cmp; at different source
# paths the bytes are expected to differ, see R6.md).
# Usage (repository root): codecmp.sh <base-build-dir> <new-build-dir>
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
objcmp=$here/objcmp.py objdiff=$here/../R4-bench/objdiff.py sasscmp=$here/../C5-bench/sass_cmp.py
base=$1 new=$2 bad=0 objects=0 functions=0 ws=0
while read -r rel; do
    objects=$((objects + 1))
    if [[ ! -f $new/$rel ]]; then echo "missing in new: $rel"; bad=1; continue; fi
    out=$(python3 "$objcmp" "$base/$rel" "$new/$rel") || { echo "host DIFFER: $rel"; bad=1; }
    echo "$out" | grep -q '^strings-ws\|^DIFFER' && { echo "$rel:"; echo "$out" | sed 's/^/  /'; }
    functions=$((functions + $(echo "$out" | sed -n 's/^\([0-9]*\) \/ .*/\1/p')))
    ws=$((ws + $(echo "$out" | grep -c '^strings-ws')))
done < <(cd "$base" && find . -name '*.o' -path '*CMakeFiles*' | sed 's|^\./||' | sort)
echo "host: $objects object files, $functions functions: $( ((bad)) && echo DIFFER || echo "identical instructions and data sections; $ws functions reference stringified test expressions that differ only in whitespace")"
for lib in libkan.a libkan_cuda.a python/_kan*.so; do
    for f in "$base"/$lib; do
        [[ -f $f ]] || continue
        rel=${f#"$base"/}
        out=$(python3 "$objdiff" lib "$f" "$new/$rel") || bad=1
        echo "$rel: $(echo "$out" | tail -n 1)$(echo "$out" | grep -q '^differs\|^only' && echo ' DIFFER')"
        # Bytes: identical, or the NUL-delimited strings that contain every differing byte
        # (at the same source path: only nvcc's temporary file name, the STT_FILE symbol).
        if cmp -s "$f" "$new/$rel"; then echo "  bytes: identical"
        else python3 - "$f" "$new/$rel" <<'PY'
import sys
a, b = (open(p, 'rb').read() for p in sys.argv[1:3])
if len(a) != len(b):
    print(f'  bytes: sizes differ ({len(a)} vs {len(b)})')
else:
    names = set()
    for k in (k for k in range(len(a)) if a[k] != b[k]):
        names.add(a[a.rfind(b'\0', 0, k) + 1:a.find(b'\0', k)].decode(errors='replace'))
    n = sum(x != y for x, y in zip(a, b))
    if len(names) <= 8:
        print(f'  bytes: {n} differ, all within the strings', sorted(names))
    else:
        print(f'  bytes: {n} differ (layout differs; no string list)')
PY
        fi
    done
done
while read -r rel; do
    echo "device $rel:"
    python3 "$sasscmp" "$base/$rel" "$new/$rel" | sed 's/^/  /'
    python3 "$objdiff" sass "$base/$rel" "$new/$rel" | sed 's/^/  whole text: /' || bad=1
    python3 "$sasscmp" "$base/$rel" "$new/$rel" | grep -q 'differing 0$' || bad=1
done < <(cd "$base" && find . -name '*.cu.o' | sed 's|^\./||' | sort)
exit $bad
