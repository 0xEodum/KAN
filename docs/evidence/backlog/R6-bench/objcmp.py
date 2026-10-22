#!/usr/bin/env python3
"""Backlog R6: host code of two relocatable objects, function by function.

  objcmp.py <before.o> <after.o>

Like R4's objdiff.py (objdump -dr, addresses and the hash nvcc derives from
the source path, in `_GLOBAL__N__<hash>_` and `_INTERNAL_<hash>_` names,
normalized), but a relocation into a string-literal pool
(`.rodata[.<function>].str1.N+0xOFF`) is replaced by the NUL-terminated string at that
offset, read from the object's own section. A reformat moves pooled strings
(the test macros stringify their expressions: `#expression`), which changes
those offsets but no instruction. Per function the result is:
  identical      - same instructions and the same referenced strings;
  strings-ws     - same instructions; referenced strings equal after removing
                   whitespace (a reformatted stringified expression);
  DIFFER         - anything else.
Data: every `.rodata*` (except the string pools; nvcc's path hash normalized in
names and bytes), `.data*` and CUDA fatbinary
(`.nv_fatbin`, `.nvFatBinSegment`) section is compared byte for byte, so constant
pools, tables and embedded device code are covered too.
Prints one line per non-identical function or data section and a summary; exit 1
on DIFFER.
"""
import re
import subprocess
import sys
import struct

ANON = re.compile(r'_GLOBAL__N__[0-9a-f]{8}_|_INTERNAL_[0-9a-f]{8}_')
POOL = re.compile(r'(\.rodata(?:\.[^\s+]*?)?\.str[0-9]+\.[0-9]+)(?:\+0x([0-9a-f]+))?')
ANON_BYTES = re.compile(rb'_GLOBAL__N__[0-9a-f]{8}_|_INTERNAL_[0-9a-f]{8}_')


def sections(obj):
    """{name: [bytes, ...]} of an ELF64 little-endian relocatable object (SHT_NOBITS: empty)."""
    with open(obj, 'rb') as f:
        data = f.read()
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    heads = [struct.unpack_from('<IIQQQQ', data, shoff + i * shentsize) for i in range(shnum)]
    names = heads[shstrndx]
    result = {}
    for name, kind, _, _, offset, size in heads:
        end = data.index(b'\0', names[4] + name)
        label = data[names[4] + name:end].decode()
        result.setdefault(label, []).append(b'' if kind == 8 else data[offset:offset + size])
    return result


DATA = re.compile(r'^\.(rodata|data|nv_fatbin|nvFatBinSegment)')


def data_sections(table):
    """Constant and initialized data, compared byte for byte (string pools are compared
    through the instructions that reference them)."""
    result = {}
    for k, v in table.items():
        if DATA.match(k) and not POOL.fullmatch(k):
            # nvcc's path-derived hash in section names and in embedded names (same length).
            result[ANON.sub('_H_', k)] = [ANON_BYTES.sub(b'_H_', x) for x in v]
    return result


def functions(obj):
    text = subprocess.run(['objdump', '-dr', '--no-show-raw-insn', obj],
                          capture_output=True, text=True, check=True).stdout
    table, funcs, cur = sections(obj), {}, None

    def resolve(m):
        data = table[m.group(1)][0]
        off = int(m.group(2) or '0', 16)
        end = data.find(b'\0', off)
        return 'str' + repr(data[off:end if end >= 0 else len(data)].decode('latin-1'))

    for line in text.splitlines():
        head = re.match(r'^[0-9a-f]+ <(.+)>:$', line)
        if head:
            cur = ANON.sub('_H_', head.group(1))
            cur = re.sub(r'_GLOBAL__sub_I_.*', '_GLOBAL__sub_I_<file>', cur)
            funcs.setdefault(cur, [])
            continue
        if 'file format' in line:
            cur = None
            continue
        if cur is None or not line.strip() or line.startswith('Disassembly'):
            continue
        line = re.sub(r'^\s*[0-9a-f]+:\s*', '', line)
        line = re.sub(r'\b[0-9a-f]+ <', '<', line)
        funcs[cur].append(ANON.sub('_H_', POOL.sub(resolve, line)))
    return funcs, data_sections(table)


def squeeze(lines):
    return [re.sub(r"str'.*'", lambda m: re.sub(r'\s', '', m.group(0)), x) for x in lines]


def main():
    (a, da), (b, db) = functions(sys.argv[1]), functions(sys.argv[2])
    counts = {'identical': 0, 'strings-ws': 0, 'DIFFER': 0}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print('DIFFER only in', 'after' if name in b else 'before', name)
            counts['DIFFER'] += 1
            continue
        kind = 'identical' if a[name] == b[name] else 'strings-ws' if squeeze(a[name]) == squeeze(b[name]) else 'DIFFER'
        counts[kind] += 1
        if kind == 'strings-ws':
            x, y = next((x, y) for x, y in zip(a[name], b[name]) if x != y)
            print(f'{kind} {name}: e.g. {re.search(r"str.*", x).group(0)[:70]} -> {re.search(r"str.*", y).group(0)[:70]}')
        elif kind != 'identical':
            print(f'{kind} {name}')
    data_bad = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    for k in data_bad:
        print('DIFFER data section', k)
    print(f'{len(a)} / {len(b)} functions:', ', '.join(f'{v} {k}' for k, v in counts.items()),
          f'| {len(da)} data sections ({sum(len(x) for v in da.values() for x in v)} bytes):',
          'identical' if not data_bad else f'{len(data_bad)} DIFFER')
    return 1 if counts['DIFFER'] or data_bad else 0


sys.exit(main())
