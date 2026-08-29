"""
coverage_audit.py — find harvesting gaps before spending enrichment time on a domain.

For each domain, imports the real library, pulls its public callables via
introspection, and reports what fraction actually made it into trees/{domain}.json.
This is meant to answer "which trees need a real re-harvest" without hand-picking
functions from a failing demo query — run it, then re-run the harvester (not this
script) against whatever modules it flags as low-coverage.

This only checks presence, not quality — a function showing up as "covered" can
still have the empty-docstring/garbled-tag problems from before. It's a triage
step, not a replacement for the enrichment pass.
"""
import importlib
import inspect
import json
from pathlib import Path

# domain -> (module(s) to introspect, min expected coverage before flagging)
TARGETS = {
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "matplotlib": ["matplotlib.pyplot"],
    "sklearn": ["sklearn.preprocessing", "sklearn.linear_model", "sklearn.model_selection"],
    "cv2": ["cv2"],
    "scipy": ["scipy.stats", "scipy.optimize"],
}

TREES_DIR = Path("trees")


def public_callables(module_name: str) -> set[str]:
    mod = importlib.import_module(module_name)
    names = set()
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if callable(obj):
            names.add(name.lower())
    return names


def harvested_names(domain: str) -> set[str]:
    """Tokenized, not substring — 'matplotlib' contains the literal substring
    'plot' (mat-PLOT-lib), which silently inflated coverage in an earlier version
    of this script. Split into whole word-ish tokens and require exact match."""
    import re
    path = TREES_DIR / f"{domain}.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    tokens: set[str] = set()
    for cell in data.get("cells", []):
        blob = " ".join([
            cell.get("cell_id", ""),
            cell.get("code_template", ""),
            " ".join(cell.get("dependencies", [])),
        ]).lower()
        tokens |= set(re.findall(r"[a-z][a-z0-9_]*", blob))
    return tokens


def main():
    for domain, modules in TARGETS.items():
        expected: set[str] = set()
        for m in modules:
            try:
                expected |= public_callables(m)
            except ImportError:
                print(f"[{domain}] could not import {m} (not installed here) — skipping")
                continue

        haystack = harvested_names(domain)
        missing = [fn for fn in sorted(expected) if fn not in haystack]

        pct_covered = 100 * (len(expected) - len(missing)) / max(len(expected), 1)
        print(f"\n[{domain}] {len(expected)} public callables checked | "
              f"{pct_covered:.0f}% appear present | {len(missing)} likely missing")
        if missing:
            sample = missing[:15]
            print(f"  sample missing: {sample}" + (" ..." if len(missing) > 15 else ""))


if __name__ == "__main__":
    main()
