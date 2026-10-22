#!/usr/bin/env bash
# Source format check (backlog R6): runs clang-format 18 with the repository's
# .clang-format over the formatted scope and fails if any file would change.
# Scope: C/C++/CUDA sources under include/, src/, tests/, benchmarks/, examples/
# and python/ (docs/ evidence harnesses are excluded: docs/.clang-format).
# Tool: $CLANG_FORMAT, else .venv/bin/clang-format, else clang-format-18 or
# clang-format on PATH. Pinned version 18.1.8 (`pip install -r requirements-dev.txt`);
# another major version formats differently, so it is reported and the script
# exits 77 (CTest SKIP_RETURN_CODE), as it does when no clang-format is found or
# bash is older than 4. With CHECK_FORMAT_STRICT=1 (CI) those cases fail instead.
# Usage (any directory): scripts/check_format.sh [--fix]   (--fix rewrites the files)
set -uo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
skip() { # reason
    echo "check_format: $1; skipped" >&2
    [[ ${CHECK_FORMAT_STRICT:-0} == 1 ]] && { echo "check_format: CHECK_FORMAT_STRICT=1, failing" >&2; exit 1; }
    exit 77
}
fix=0
case ${1:-} in
    --fix) fix=1 ;;
    "") ;;
    *) echo "usage: scripts/check_format.sh [--fix]" >&2; exit 2 ;;
esac
((BASH_VERSINFO[0] >= 4)) || skip "bash ${BASH_VERSION} is older than 4"
tool=${CLANG_FORMAT:-}
if [[ -z $tool ]]; then
    for candidate in "$root/.venv/bin/clang-format" clang-format-18 clang-format; do
        if command -v "$candidate" > /dev/null 2>&1; then tool=$candidate; break; fi
    done
fi
if [[ -z $tool ]] || ! command -v "$tool" > /dev/null 2>&1; then
    skip "clang-format not found (pip install -r requirements-dev.txt)"
fi
version=$("$tool" --version | sed -n 's/.*clang-format version \([0-9][0-9.]*\).*/\1/p')
if [[ ${version%%.*} != 18 ]]; then
    skip "$tool is version '${version:-unknown}', the pinned major version is 18"
fi
cd "$root" || exit 2
if [[ ! -f .clang-format ]] || ! "$tool" --style=file --dump-config src/core/layer.cpp > /dev/null 2>&1; then
    echo "check_format: $root/.clang-format is missing or invalid for clang-format $version" >&2
    exit 2
fi
scope=(include src tests benchmarks examples python)
pattern='\.(c|cc|cpp|cxx|h|hh|hpp|hxx|cu|cuh)$'
files=()
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    # Tracked and new (untracked, not ignored) files that exist in the working tree.
    while IFS= read -r f; do [[ -f $f ]] && files+=("$f"); done < <(
        git ls-files --cached --others --exclude-standard -- "${scope[@]}" | grep -E "$pattern" | sort -u)
else
    while IFS= read -r f; do files+=("$f"); done < <(find "${scope[@]}" -type f 2> /dev/null | grep -E "$pattern" | sort)
fi
if ((${#files[@]} == 0)); then echo "check_format: no source files found" >&2; exit 2; fi
if ((fix)); then
    "$tool" -i --style=file "${files[@]}" || exit 1
    echo "check_format: formatted ${#files[@]} files with clang-format $version"
    exit 0
fi
if "$tool" --dry-run --Werror --style=file "${files[@]}"; then
    echo "check_format: ${#files[@]} files formatted (clang-format $version)"
else
    echo "check_format: files above need formatting; run scripts/check_format.sh --fix" >&2
    exit 1
fi
