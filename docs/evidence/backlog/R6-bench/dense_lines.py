#!/usr/bin/env python3
"""Backlog R6: count lines holding more than one statement (the R6 finding).

  dense_lines.py <rev> <file>...    (repository root; <rev> = WORKTREE for the working tree)

Heuristic per line: string/char literals and // comments removed; parenthesized
groups (for headers, if/switch initializers, arguments, so also lambdas passed as
arguments; a group continued on the next line is cut), braced initializer lists and
one-statement brace bodies (`{ return x; }`: short functions and lambdas, kept on
one line by the AllowShort* options) collapsed; the line counts as dense when a `;`
is followed by more code on the same line. `case`/`default`
labels are skipped (short dispatch cases such as `case 1: f(x); break;` stay on
one line by AllowShortCaseLabelsOnASingleLine).
"""
import re
import subprocess
import sys

STR = re.compile(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'')


def dense(line):
    code = STR.sub('""', line).split('//')[0].strip()
    if not code or code.startswith(('case ', 'default:')):
        return False
    while True:  # parenthesized groups: for headers, if/switch initializers, call arguments
        collapsed = re.sub(r'\([^()]*\)', '()', code)
        if collapsed == code:
            break
        code = collapsed
    if '(' in code:  # a group continued on the next line
        code = code[:code.index('(')]
    while True:  # braced lists, then one-statement bodies ({ return x; })
        collapsed = re.sub(r'\{(?:[^{};]|\{\})*;?\s*\}', '{}', code)
        if collapsed == code:
            break
        code = collapsed
    parts = code.split(';')
    return any(part.strip() not in ('', '}', '};') for part in parts[1:])


def main():
    rev, files = sys.argv[1], sys.argv[2:]
    total = 0
    for name in files:
        text = open(name).read() if rev == 'WORKTREE' else subprocess.run(
            ['git', 'show', f'{rev}:{name}'], capture_output=True, text=True, check=True).stdout
        total += sum(dense(line) for line in text.splitlines())
    print(f'{rev}: {total} lines with more than one statement in {len(files)} files')


main()
