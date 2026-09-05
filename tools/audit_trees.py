"""
audit_trees.py — structural QA gate for trees/*.json before promoting an
enrichment checkpoint (or any harvest run) into the live tree files.

Five independent checks, each answering a different question:

1. SCHEMA FINGERPRINTING (the thing you asked for): every cell's shape —
   which top-level keys it has, and which keys each of its ports has — gets
   hashed into a fingerprint. Cells are grouped by fingerprint. A schema that's
   shared by thousands of cells is almost certainly fine; a schema shared by a
   handful is worth a human glancing at, since it means those cells were built
   or touched by a different code path than everything else.

2. DOCSTRING COMPLETENESS — cells with no docstring at all (should be near-zero
   after enrichment) and, separately, cells whose docstring is character-for-
   character IDENTICAL to another cell's — a known LLM-batch failure mode where
   a generic filler description gets silently reused instead of one specific
   per-cell answer.

3. TYPE CONSISTENCY — for every port, checks that `default_value`'s actual JSON
   type matches what `type_name` claims (e.g. type_name="bool" but
   default_value is the string "True", not JSON true).

4. TEMPLATE WIRING — cross-checks every `{placeholder}` in `code_template`
   against the cell's declared `inputs` (plus the always-valid `output_var`,
   `dest_path`, `filepath`), in both directions: a placeholder with no matching
   input (would KeyError at synthesis time), and a declared input that's never
   referenced in the template (dead/unused port).

5. DUPLICATE CELL_IDs — should be structurally impossible given dict-keyed
   merges, but costs nothing to confirm.

Usage:
    python3 audit_trees.py trees/pandas.json
    python3 audit_trees.py nstl_enrichment/checkpoints/pandas.json
    python3 audit_trees.py trees/*.json          # whole corpus at once
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RESERVED_PLACEHOLDERS = {"output_var"}


def port_fingerprint(port: Dict[str, Any]) -> tuple:
    return tuple(sorted(port.keys()))


def cell_fingerprint(cell: Dict[str, Any]) -> tuple:
    top_keys = tuple(sorted(cell.keys()))
    input_shapes = tuple(sorted(port_fingerprint(p) for p in cell.get("inputs", {}).values()))
    output_shapes = tuple(sorted(port_fingerprint(p) for p in cell.get("outputs", {}).values()))
    return (top_keys, input_shapes, output_shapes)


def check_schema_fingerprints(cells: List[Dict[str, Any]]) -> Dict[tuple, List[str]]:
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for c in cells:
        groups[cell_fingerprint(c)].append(c["cell_id"])
    return groups


def check_docstrings(cells: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    empty = [c["cell_id"] for c in cells if not c.get("docstring")]
    by_text: Dict[str, List[str]] = defaultdict(list)
    for c in cells:
        doc = c.get("docstring")
        if doc:
            by_text[doc].append(c["cell_id"])
    # only text shared by cells that are NOT trivially the same function name pattern
    suspicious_duplicates = {doc: ids for doc, ids in by_text.items() if len(ids) > 1}
    return {"empty": empty, "duplicated_text": suspicious_duplicates}


PY_TYPE_CHECKS = {
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


def check_type_consistency(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    problems = []
    for c in cells:
        for direction in ("inputs", "outputs"):
            for pname, port in c.get(direction, {}).items():
                default = port.get("default_value")
                type_name = port.get("type_name")
                if default is None or type_name not in PY_TYPE_CHECKS:
                    continue
                if not PY_TYPE_CHECKS[type_name](default):
                    problems.append({
                        "cell_id": c["cell_id"],
                        "port": f"{direction}.{pname}",
                        "type_name": type_name,
                        "default_value": default,
                        "actual_json_type": type(default).__name__,
                    })
    return problems


def check_template_wiring(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    problems = []
    for c in cells:
        template = c.get("code_template", "")
        placeholders = set(re.findall(r"\{(\w+)\}", template))
        declared = set(c.get("inputs", {}).keys()) | RESERVED_PLACEHOLDERS
        # dest_path/filepath show up as their own input keys already if declared —
        # only flag as "unmatched" if truly absent from inputs and not reserved.
        unmatched = placeholders - declared
        unused = declared - placeholders - RESERVED_PLACEHOLDERS
        if unmatched or unused:
            problems.append({
                "cell_id": c["cell_id"],
                "template": template,
                "unmatched_placeholders": sorted(unmatched),
                "unused_declared_inputs": sorted(unused),
            })
    return problems


def check_port_descriptions(cells: List[Dict[str, Any]]) -> Dict[str, int]:
    """Informational only — not a functional bug. Tracked for visibility, not
    a blocker: nothing in synthesis reads port-level `description`, and it's
    been empty since the original harvester, well before any enrichment work."""
    total = empty = 0
    for c in cells:
        for port in list(c.get("inputs", {}).values()) + list(c.get("outputs", {}).values()):
            total += 1
            if not port.get("description"):
                empty += 1
    return {"total_ports": total, "empty_description": empty}


def check_duplicate_ids(cells: List[Dict[str, Any]]) -> List[str]:
    seen = defaultdict(int)
    for c in cells:
        seen[c["cell_id"]] += 1
    return [cid for cid, n in seen.items() if n > 1]


def audit_file(path: Path) -> None:
    data = json.loads(path.read_text())
    cells = data.get("cells", [])
    print(f"\n{'='*70}\n{path} — {len(cells)} cells\n{'='*70}")

    # 1. Schema fingerprints
    groups = check_schema_fingerprints(cells)
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    print(f"\n[schema fingerprints] {len(groups)} distinct shapes found")
    for fp, ids in sorted_groups[:5]:
        print(f"  {len(ids):6d} cells  <- dominant shape(s), e.g. {ids[0]}")
    minority = [(fp, ids) for fp, ids in sorted_groups if len(ids) <= 3]
    if minority:
        print(f"  {len(minority)} shapes used by 3 or fewer cells (worth a manual look):")
        for fp, ids in minority[:15]:
            print(f"    {ids} -> top_keys={fp[0]}")

    # 2. Docstrings
    doc_report = check_docstrings(cells)
    print(f"\n[docstrings] empty: {len(doc_report['empty'])}")
    if doc_report["empty"]:
        print(f"  {doc_report['empty'][:10]}")
    dup_text = doc_report["duplicated_text"]
    if dup_text:
        print(f"  {len(dup_text)} docstring texts reused across multiple cells:")
        for doc, ids in list(dup_text.items())[:5]:
            print(f"    '{doc[:60]}...' used by {len(ids)} cells: {ids[:5]}")

    # 3. Type consistency
    type_problems = check_type_consistency(cells)
    print(f"\n[type consistency] {len(type_problems)} default_value/type_name mismatches")
    for p in type_problems[:10]:
        print(f"  {p['cell_id']} | {p['port']} | type_name={p['type_name']} "
              f"but default_value={p['default_value']!r} ({p['actual_json_type']})")

    # 4. Template wiring
    wiring_problems = check_template_wiring(cells)
    print(f"\n[template wiring] {len(wiring_problems)} cells with placeholder mismatches")
    for p in wiring_problems[:10]:
        print(f"  {p['cell_id']}: unmatched={p['unmatched_placeholders']} "
              f"unused={p['unused_declared_inputs']} | template={p['template'][:60]}")

    # 5. Duplicate IDs
    dupes = check_duplicate_ids(cells)
    print(f"\n[duplicate cell_ids] {len(dupes)}")
    if dupes:
        print(f"  {dupes}")

    # 6. Port descriptions (informational only, not a functional blocker)
    desc = check_port_descriptions(cells)
    pct = 100 * desc["empty_description"] / max(desc["total_ports"], 1)
    print(f"\n[port descriptions - informational, not a blocker] "
          f"{desc['empty_description']}/{desc['total_ports']} empty ({pct:.0f}%)")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["trees/pandas.json"]
    for p in paths:
        audit_file(Path(p))
