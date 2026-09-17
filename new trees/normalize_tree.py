#!/usr/bin/env python3
"""
normalize_tree.py - Post-process an LLM-generated NSTL domain tree.

What this does, and why:

1. VALIDATES structural integrity that a single LLM generation pass over
   ~350 nodes is likely to violate:
     - Dangling edges[].target_cell_id (references a cell that was never
       generated, e.g. truncated near a context limit)
     - Dangling transitions[].via (same problem, at the typestate level)
     - typestates.states[].parent_state referencing an undefined state name
     - Duplicate cell_id values

2. CROSS-CHECKS the hand-authored `typestates.transitions` array against
   what is mechanically derivable from each cell's own input/output port
   `state` fields. `transitions` should be a *build artifact*, not
   hand-authored data — this reports every mismatch so you can see whether
   the LLM invented a transition that doesn't match any real cell, or
   missed one that a cell clearly implements.

3. NORMALIZES `edges[].affinity_score`. LLM-invented floats like 0.95 vs
   0.85 encode false precision (no real usage measurement backs the
   specific number), but the LLM's *relative ranking* within one node's own
   edge list is usually a decent reflection of real-world idiom frequency
   for a well-documented library like cv2. So instead of keeping or
   discarding the numbers wholesale, this re-buckets each node's edges by
   rank into a small number of honest tiers, and tags every edge with a
   `score_provenance` field so nothing here is silently treated as
   measured ground truth later.

   Strategies (--strategy):
     rank_bucket (default) - keep relative order, discard fake precision
     null                  - strip scores entirely (score -> null)
     keep                  - leave scores untouched, only add provenance tag

Usage:
    python normalize_tree.py input_tree.json output_tree.json \
        [--strategy rank_bucket|null|keep] [--tiers 0.8,0.5,0.2]
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple


def load_tree(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cell_id_set(tree: Dict[str, Any]) -> List[str]:
    return [c["cell_id"] for c in tree.get("cells", [])]


def state_name(s: Any) -> Optional[str]:
    """Typestate entries can be either a full object or a bare string per
    the schema ("states": Array of TypestateDefinition objects or
    strings) — this handles both without assuming the richer form."""
    if isinstance(s, dict):
        return s.get("name")
    if isinstance(s, str):
        return s
    return None


def find_duplicate_cell_ids(cell_ids: List[str]) -> List[str]:
    seen, dupes = set(), []
    for cid in cell_ids:
        if cid in seen:
            dupes.append(cid)
        seen.add(cid)
    return dupes


def iter_ports(ports: Any) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    """Normalize a cell's inputs/outputs field into (name, port_dict) pairs.
    Per the schema this should always be an object mapping port_name ->
    PortSchema, but LLM generations sometimes drift into emitting a bare
    list of port objects instead (seen in practice on multi-output
    cells). Handles both without crashing; list-form ports fall back to
    a "name" key inside the port object itself, or None if absent."""
    if isinstance(ports, dict):
        return list(ports.items())
    if isinstance(ports, list):
        pairs = []
        for p in ports:
            if isinstance(p, dict):
                pairs.append((p.get("name") or p.get("port_name"), p))
        return pairs
    return []


def _pick_primary_port(ports: Any, prefer_non_scalar: bool = False) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Best-effort pick of 'the' primary port from a cell's inputs or outputs
    (dict OR list form, see iter_ports), mirroring the spirit of
    CellSchema.primary_input/primary_output without needing the full
    pydantic model loaded. Always prefers a non-scalar port when
    available — multi-output cells (e.g. cv2.threshold returning
    (retval, dst)) commonly list a scalar diagnostic first, and naively
    taking order[0] silently grabs the wrong one. Returns (name, port)."""
    pairs = iter_ports(ports)
    if not pairs:
        return None, None
    items = [p for _, p in pairs]
    pool_pairs = pairs
    required_pairs = [(n, p) for n, p in pairs if p.get("required")] or pairs
    non_scalar_pairs = [(n, p) for n, p in required_pairs if (p.get("abstract_type") or "").lower()
                         not in ("scalar", "text", "path", "logical")]
    pool_pairs = non_scalar_pairs or required_pairs
    return pool_pairs[0]


def primary_input_states(cell: Dict[str, Any]) -> List[str]:
    """All states a cell's primary input port accepts: its declared `state`
    plus any listed in an `accepted_states` list (for genuinely polymorphic
    functions like GaussianBlur, which works on both gray and color)."""
    _, port = _pick_primary_port(cell.get("inputs", {}), prefer_non_scalar=True)
    if not port:
        return []
    states = []
    if port.get("state"):
        states.append(port["state"])
    for s in port.get("accepted_states", []) or []:
        if s not in states:
            states.append(s)
    return states


def primary_output_state(cell: Dict[str, Any]) -> Optional[str]:
    _, port = _pick_primary_port(cell.get("outputs", {}), prefer_non_scalar=False)
    return port.get("state") if port else None


def derive_transitions(tree: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Mechanically derive (from_state, to_state, via_cell_id) triples from
    each cell's own declared port states. This is the single source of
    truth `typestates.transitions` should be generated from. A cell with
    multiple accepted input states (polymorphic over typestate) emits one
    triple per accepted state."""
    derived = []
    for cell in tree.get("cells", []):
        out_state = primary_output_state(cell)
        if not out_state:
            continue
        for in_state in primary_input_states(cell):
            derived.append((in_state, out_state, cell["cell_id"]))
    return derived


def is_generic(transition: Tuple[str, str, str]) -> bool:
    """True for transitions that don't represent a meaningful state change
    worth tracking (e.g. any -> any arithmetic/utility ops)."""
    f, t, _via = transition
    return f == "any" or t == "any"


def normalize_port_shapes(tree: Dict[str, Any]) -> List[Tuple[str, str, int]]:
    """Converts any cell whose `inputs`/`outputs` drifted into list-form
    (schema says these should be an object mapping port_name -> PortSchema)
    back into proper dict-form, in place. This isn't just for this script's
    own analysis — the actual engine's port loader expects a dict too, so
    list-form ports would silently break downstream regardless. Returns a
    list of (cell_id, direction, port_count) for every cell that needed
    fixing, so it can be reported rather than fixed silently."""
    fixed = []
    for cell in tree.get("cells", []):
        for direction in ("inputs", "outputs"):
            raw = cell.get(direction)
            if isinstance(raw, list):
                pairs = iter_ports(raw)
                new_dict = {}
                for i, (name, port) in enumerate(pairs):
                    key = name or f"port_{i}"
                    new_dict[key] = port
                cell[direction] = new_dict
                fixed.append((cell.get("cell_id", "?"), direction, len(pairs)))
    return fixed


def normalize_tree_metadata(tree: Dict[str, Any]) -> None:
    """Ensures root metadata fields and port roles conform to schema."""
    if "type_vars" not in tree:
        tree["type_vars"] = []
    if "aliases" not in tree:
        tree["aliases"] = {}
    if "top_types" not in tree:
        tree["top_types"] = []
    if "product_constructors" not in tree:
        tree["product_constructors"] = []

    # Normalize per-port role / port_role across all cells
    for cell in tree.get("cells", []):
        for direction in ("inputs", "outputs"):
            ports = cell.get(direction)
            if isinstance(ports, dict):
                for p_name, p_data in ports.items():
                    if isinstance(p_data, dict):
                        role = p_data.get("role") or p_data.get("port_role")
                        if role:
                            p_data["role"] = str(role).strip().lower()
                            p_data["port_role"] = str(role).strip().lower()


def validate(tree: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "duplicate_cell_ids": [],
        "dangling_edge_targets": [],       # (source_cell_id, target_cell_id)
        "dangling_transition_via": [],     # (from, to, via)
        "orphan_parent_states": [],        # (state_name, missing_parent)
        "transition_mismatches": {
            "in_llm_not_derivable": [],        # LLM wrote it, no matching cell found — needs a human look
            "derivable_not_in_llm_real": [],   # a cell implies it (non-generic), LLM omitted it
            "derivable_not_in_llm_generic": [],# a cell implies it but it's any->any noise — ignore
        },
        "undeclared_states": [],  # (cell_id, port_name, state_name) using a state never in typestates.states
        "malformed_transitions": [],  # raw entries in typestates.transitions missing from/to/via
    }

    cell_ids = get_cell_id_set(tree)
    cell_id_set = set(cell_ids)
    report["duplicate_cell_ids"] = find_duplicate_cell_ids(cell_ids)

    # Dangling edge targets
    for cell in tree.get("cells", []):
        for edge in cell.get("edges", []) or []:
            target = edge.get("target_cell_id")
            if target and target not in cell_id_set:
                report["dangling_edge_targets"].append((cell["cell_id"], target))

    # Dangling / orphan typestate references
    typestates = tree.get("typestates", {})
    state_names = {state_name(s) for s in typestates.get("states", [])}
    state_names.discard(None)
    for s in typestates.get("states", []):
        if not isinstance(s, dict):
            continue  # bare-string states have no parent_state to check
        parent = s.get("parent_state")
        if parent and parent not in state_names:
            report["orphan_parent_states"].append((s.get("name"), parent))

    llm_transitions = set()
    for t in typestates.get("transitions", []) or []:
        if not isinstance(t, dict) or not all(k in t for k in ("from", "to", "via")):
            report["malformed_transitions"].append(t)
            continue
        llm_transitions.add((t["from"], t["to"], t["via"]))
    for (f, t, via) in llm_transitions:
        if via not in cell_id_set:
            report["dangling_transition_via"].append((f, t, via))

    derived = set(derive_transitions(tree))
    report["transition_mismatches"]["in_llm_not_derivable"] = sorted(
        llm_transitions - derived
    )
    missing = derived - llm_transitions
    report["transition_mismatches"]["derivable_not_in_llm_real"] = sorted(
        t for t in missing if not is_generic(t)
    )
    report["transition_mismatches"]["derivable_not_in_llm_generic"] = sorted(
        t for t in missing if is_generic(t)
    )

    # Undeclared states: every state string used on a cell's *primary* port
    # (the one that actually participates in the pipeline/transition graph,
    # same port derive_transitions() reads) should be registered in
    # typestates.states. Secondary ports — keyword parameters like
    # kernel_size, sigma, flags, delay_ms — are a different thing entirely
    # (parameter-role labels for future slot binding, not pipeline
    # typestates) and are intentionally NOT checked here.
    for cell in tree.get("cells", []):
        inputs = cell.get("inputs", {}) or {}
        outputs = cell.get("outputs", {}) or {}
        in_name, in_port = _pick_primary_port(inputs, prefer_non_scalar=True)
        out_name, out_port = _pick_primary_port(outputs, prefer_non_scalar=False)
        candidates = []  # (port_name, state)
        if in_port:
            if in_port.get("state"):
                candidates.append((in_name or "?", in_port["state"]))
            for s in in_port.get("accepted_states", []) or []:
                candidates.append((in_name or "?", s))
        if out_port:
            if out_port.get("state"):
                candidates.append((out_name or "?", out_port["state"]))
        for port_name, s in candidates:
            if s not in state_names and s not in ("any", "default"):
                report["undeclared_states"].append((cell["cell_id"], port_name, s))

    return report


def normalize_scores(
    tree: Dict[str, Any],
    strategy: str = "rank_bucket",
    tiers: Tuple[float, float, float] = (0.8, 0.5, 0.2),
) -> None:
    """Mutates tree in place: tags provenance, and rewrites affinity_score
    per the chosen strategy."""
    for cell in tree.get("cells", []):
        edges = cell.get("edges", []) or []
        if not edges:
            continue

        if strategy == "keep":
            for e in edges:
                e["score_provenance"] = e.get("score_provenance", "llm_seed")
                e["needs_mining"] = True
            continue

        if strategy == "null":
            for e in edges:
                e["affinity_score"] = None
                e["score_provenance"] = "unscored"
                e["needs_mining"] = True
            continue

        # rank_bucket (default): preserve relative order the LLM gave us,
        # discard the fake-precision absolute value.
        ranked = sorted(
            edges, key=lambda e: e.get("affinity_score") or 0.0, reverse=True
        )
        n = len(ranked)
        for i, e in enumerate(ranked):
            if n <= 1:
                tier = tiers[0]
            elif i < max(1, n // 3):
                tier = tiers[0]
            elif i < max(2, 2 * n // 3):
                tier = tiers[1]
            else:
                tier = tiers[2]
            e["affinity_score"] = tier
            e["score_provenance"] = "llm_seed"
            e["needs_mining"] = True


def print_report(report: Dict[str, Any]) -> None:
    def _n(x):
        return len(x) if isinstance(x, (list, set)) else x

    print("=== Tree Validation Report ===")
    print(f"Duplicate cell_ids:                 {_n(report['duplicate_cell_ids'])}")
    if report["duplicate_cell_ids"]:
        print(f"    {report['duplicate_cell_ids']}")

    print(f"Malformed typestates.transitions entries (missing from/to/via, skipped): "
          f"{_n(report['malformed_transitions'])}")
    for entry in report["malformed_transitions"][:10]:
        print(f"    {entry}")

    print(f"Dangling edge targets:               {_n(report['dangling_edge_targets'])}")
    for src, tgt in report["dangling_edge_targets"][:20]:
        print(f"    {src} -> {tgt}  (target cell not found)")

    print(f"Orphan parent_state references:      {_n(report['orphan_parent_states'])}")
    for name, parent in report["orphan_parent_states"]:
        print(f"    state '{name}' declares parent_state '{parent}' (undefined)")

    print(f"Dangling transitions[].via:           {_n(report['dangling_transition_via'])}")
    for f, t, via in report["dangling_transition_via"][:20]:
        print(f"    {f} -> {t} via {via}  (cell not found)")

    mism = report["transition_mismatches"]
    print(f"LLM-authored transitions with no matching cell (NEEDS A HUMAN LOOK): {_n(mism['in_llm_not_derivable'])}")
    for f, t, via in mism["in_llm_not_derivable"][:30]:
        print(f"    claimed: {f} -> {t} via {via}")
    print(f"Cell-implied REAL transitions the LLM's own list omitted (informational only — "
          f"auto-included if you pass --rewrite-transitions, no action needed): {_n(mism['derivable_not_in_llm_real'])}")
    for f, t, via in mism["derivable_not_in_llm_real"][:30]:
        print(f"    implied: {f} -> {t} via {via}")
    print(f"Cell-implied generic (any->any) transitions missing (ignore, expected): {_n(mism['derivable_not_in_llm_generic'])}")

    print(f"Undeclared states used on ports (not in typestates.states): {_n(report['undeclared_states'])}")
    for cid, port, state in report["undeclared_states"][:30]:
        print(f"    {cid}.{port} uses state '{state}' — not registered in typestates.states")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    ap.add_argument("--strategy", choices=["rank_bucket", "null", "keep"], default="rank_bucket")
    ap.add_argument("--tiers", default="0.8,0.5,0.2",
                     help="comma-separated high,medium,low scores for rank_bucket strategy")
    ap.add_argument("--rewrite-transitions", action="store_true",
                     help="replace typestates.transitions with the mechanically derived list")
    ap.add_argument("--include-generic-transitions", action="store_true",
                     help="keep any->any transitions when rewriting (default: dropped, they carry no planning signal)")
    ap.add_argument("--strict", action="store_true",
                     help="exit non-zero if any dangling reference is found")
    args = ap.parse_args()

    tree = load_tree(args.input_path)

    normalize_tree_metadata(tree)
    port_fixes = normalize_port_shapes(tree)
    if port_fixes:
        print(f"=== Port Shape Fixes ===\nFixed {len(port_fixes)} cell(s) whose inputs/outputs "
              f"were list-form instead of the required name->port object form "
              f"(this would also have broken the actual engine's loader, not just this script):")
        for cid, direction, count in port_fixes[:20]:
            print(f"    {cid}.{direction} ({count} ports) — converted to dict form, "
                  f"synthesized port_N names where no 'name' field was present")
        print()

    report = validate(tree)
    print_report(report)

    tiers = tuple(float(x) for x in args.tiers.split(","))
    if len(tiers) != 3:
        sys.exit("--tiers must have exactly 3 comma-separated values")

    normalize_scores(tree, strategy=args.strategy, tiers=tiers)

    if args.rewrite_transitions:
        derived = derive_transitions(tree)
        kept = [t for t in derived if args.include_generic_transitions or not is_generic(t)]
        dropped = len(derived) - len(kept)
        tree.setdefault("typestates", {})["transitions"] = [
            {"from": f, "to": t, "via": via} for (f, t, via) in sorted(set(kept))
        ]
        print(f"Rewrote typestates.transitions with {len(set(kept))} derived entries "
              f"(dropped {dropped} generic any->any entries; was authored by hand before).")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)
    print(f"Wrote normalized tree to {args.output_path}")

    has_dangling = bool(
        report["dangling_edge_targets"]
        or report["dangling_transition_via"]
        or report["duplicate_cell_ids"]
    )
    if args.strict and has_dangling:
        sys.exit(1)


if __name__ == "__main__":
    main()
