#!/usr/bin/env python3
"""
scripts/check_static.py - Undefined-name gate (R6-3).

Runs pyflakes over src/, tests/ and scripts/ and fails on any `undefined name`
report. py_compile cannot see an undefined name; round 5 shipped a crash that
failed all 45 sweep runs behind exactly this class of defect (R6-1/R6-3).

Usage:
    python scripts/check_static.py

Exit codes: 0 = clean, 1 = undefined names (or pyflakes unavailable).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGETS = ["src", "tests", "scripts"]

UNDEFINED_NAME_RE = re.compile(r": undefined name '([^']+)'")

# Subprocess text output is inspected structurally via the compiled pattern
# above; scripts/ tooling may use regex (the no-regex rule applies to src/).


def main() -> int:
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        print(
            "[check_static] pyflakes is not installed; install it with:\n"
            "    python -m pip install pyflakes",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "pyflakes"] + TARGETS
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    undefined = []
    for line in output.splitlines():
        m = UNDEFINED_NAME_RE.search(line)
        if m:
            undefined.append(line.strip())

    if undefined:
        print("[check_static] FAIL — undefined names found:\n")
        for line in undefined:
            print(f"  {line}")
        print(f"\n{len(undefined)} undefined name(s). Fix these before merging.")
        return 1

    other = [l for l in output.splitlines() if l.strip()]
    print("[check_static] OK — no undefined names.")
    if other:
        print(f"[check_static] note: {len(other)} non-undefined pyflakes report(s) remain (unused imports/variables):")
        for line in other[:10]:
            print(f"  {line}")
        if len(other) > 10:
            print(f"  ... and {len(other) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
