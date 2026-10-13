#!/usr/bin/env python3
"""Backlog R4: compare generated code of two builds function by function.

  objdiff.py lib <before.a> <after.a>     host code (objdump -dr) per function
  objdiff.py sass <before.o> <after.o>    device SASS (cuobjdump -sass), whole text

Normalized away: addresses, raw bytes, archive member names, the per-file hash in
nvcc's anonymous-namespace names (_GLOBAL__N__<hash>_, derived from the source
path) and the file name in static-initializer symbols. Exit status 1 if any
function differs or exists in only one build.
"""
import re
import subprocess
import sys

ANON = re.compile(r'_GLOBAL__N__[0-9a-f]{8}_')


def functions(lib):
    text = subprocess.run(['objdump', '-dr', '--no-show-raw-insn', lib],
                          capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in text.splitlines():
        head = re.match(r'^[0-9a-f]+ <(.+)>:$', line)
        if head:
            cur = ANON.sub('_GLOBAL__N__H_', head.group(1))
            cur = re.sub(r'_GLOBAL__sub_I_.*', '_GLOBAL__sub_I_<file>', cur)
            funcs.setdefault(cur, [])
            continue
        if line.startswith('In archive') or 'file format' in line:
            cur = None
            continue
        if cur is None or not line.strip() or line.startswith('Disassembly'):
            continue
        line = re.sub(r'^\s*[0-9a-f]+:\s*', '', line)
        line = re.sub(r'\b[0-9a-f]+ <', '<', line)
        funcs[cur].append(ANON.sub('_GLOBAL__N__H_', line))
    return funcs


def main():
    mode, before, after = sys.argv[1:4]
    if mode == 'sass':
        dump = lambda o: ANON.sub('_GLOBAL__N__H_', subprocess.run(
            ['cuobjdump', '-sass', o], capture_output=True, text=True, check=True).stdout)
        a, b = dump(before), dump(after)
        kernels = a.count('Function :')
        print(f'{kernels} kernels, {len(a.splitlines())} SASS lines:', 'identical' if a == b else 'DIFFER')
        return 0 if a == b else 1
    a, b = functions(before), functions(after)
    bad = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print('only in', 'after' if name in b else 'before', name); bad = 1
        elif a[name] != b[name]:
            print(f'differs: {name} ({len(a[name])} -> {len(b[name])} instructions)'); bad = 1
    print(f'{len(a)} / {len(b)} functions compared')
    return bad


sys.exit(main())
