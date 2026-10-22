#!/usr/bin/env python3
"""Backlog R6: prove that a reformat changed only whitespace.

  tokcmp.py <base-rev> <file>...     (run from the repository root)

Each file is lexed at <base-rev> (git show) and in the working tree by the
clang raw lexer (`clang -cc1 -dump-raw-tokens`, CUDA mode for .cu/.cuh so that
`<<<`/`>>>` are single tokens). Compared per file: the sequence of
(kind, spelling) of every non-whitespace token, plus an end-of-directive
marker at each real newline that ends a preprocessor line (so no token can
move into or out of a #define/#pragma/#if). Comments are compared with their
internal whitespace collapsed (clang-format re-indents block-comment lines);
comments that are exactly `// clang-format off[...]` / `// clang-format on`
are ignored. Escaped newlines are whitespace. Inside directives two whitespace
positions are compared too: between the macro name and `(` of `#define NAME(`
(function-like vs object-like macro) and inside `#include <...>`. Not covered:
whitespace inside a macro argument that the macro stringifies (`#x`); that
changes a string, which objcmp.py's strings-ws class reports instead.
Exit status 1 on any difference.
"""
import os
import re
import subprocess
import sys
import tempfile

LOC = re.compile(r'\tLoc=<[^>\n]*>\n')
MARKER = re.compile(r'^// clang-format (off|on)\b')


def tokens(text, cuda):
    with tempfile.NamedTemporaryFile('w', suffix='.cu' if cuda else '.cpp', delete=False) as f:
        f.write(text)
        path = f.name
    try:
        cmd = ['clang', '-cc1', '-dump-raw-tokens', '-std=c++20'] + (['-x', 'cuda'] if cuda else []) + [path]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    finally:
        os.unlink(path)
    result, directive, line_start = [], False, True
    words = []  # directive tokens so far (kind, spelling, preceded by whitespace)
    space = False
    for chunk in LOC.split(out):
        if not chunk:
            continue
        kind = chunk.split(' ', 1)[0]
        spelling = chunk[len(kind) + 2:chunk.rfind("'\t")]
        if kind == 'unknown' and spelling.strip() == '':
            if '\n' in spelling:
                if directive:
                    result.append(('eod', ''))
                directive, line_start = False, True
            space = True
            continue
        if kind == 'comment':
            if MARKER.match(spelling):
                continue
            spelling = ' '.join(spelling.split())
        if kind == 'hash' and line_start:
            directive, words = True, []
        if directive:
            # Whitespace that changes meaning inside a directive: after the macro name of
            # `#define NAME(` (function-like vs object-like) and inside `#include <...>`.
            names = [w[1] for w in words]
            if names[:2] == ['#', 'define'] and len(words) == 3 and spelling == '(':
                result.append(('define-paren', 'space' if space else 'attached'))
            if names[:2] == ['#', 'include'] and len(words) >= 3 and words[2][1] == '<' \
                    and '>' not in names[2:]:
                result.append(('include-space', 'space' if space else 'none'))
            words.append((kind, spelling, space))
        line_start, space = False, False
        result.append((kind, spelling))
    if directive:
        result.append(('eod', ''))
    return result


def main():
    base, files = sys.argv[1], sys.argv[2:]
    bad = 0
    total = 0
    for name in files:
        cuda = name.endswith(('.cu', '.cuh'))
        before = subprocess.run(['git', 'show', f'{base}:{name}'], capture_output=True, text=True, check=True).stdout
        with open(name) as f:
            after = f.read()
        a, b = tokens(before, cuda), tokens(after, cuda)
        total += len(a)
        if a != b:
            bad = 1
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f'DIFFER {name}: token {i}: {a[i:i + 3]} -> {b[i:i + 3]}')
    print(f'{len(files)} files, {total} tokens compared:', 'whitespace-only' if not bad else 'TOKENS DIFFER')
    return bad


sys.exit(main())
