"""
src/row_lineage.py - Neuro-Symbolic Topological Lattice (NSTL)

Row-set lineage over a synthesized pipeline. The TREES declare which cell effects
change a table's row set (`verification_semantics.row_set_effects`); the engine only
propagates "row epochs" along data dependence and reports where a value computed on
one row set is written into a table that lives on another (e.g. an array built
before `dropna` assigned as a column of the frame after it).

No names of operations, libraries or columns appear here.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple


def _names(expr: Any, known: Dict[str, int]) -> List[str]:
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError:
        return []
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in known]


def analyze(pipeline_bindings: List[Tuple[Any, Dict[str, str]]]) -> List[Dict[str, Any]]:
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        row_effects = {str(e).lower() for e in reg.get_verification_semantics("row_set_effects")}
    except Exception:
        return []
    if not row_effects:
        return []
    epoch: Dict[str, int] = {}
    issues: List[Dict[str, Any]] = []
    for cell, bindings in pipeline_bindings:
        out = bindings.get("output_var")
        in_refs: Dict[str, List[str]] = {p: _names(v, epoch) for p, v in bindings.items() if p != "output_var"}
        # value-to-store port vs. table port of the same cell
        for p_name, port in getattr(cell, "inputs", {}).items():
            if getattr(port, "binds", None) != "assigned_value" or p_name not in in_refs:
                continue
            val_ep = max((epoch[n] for n in in_refs[p_name]), default=None)
            for q_name, q in getattr(cell, "inputs", {}).items():
                q_type = str(getattr(getattr(q, "signature", q), "type_name", "") or "").lower()
                if q_name == p_name or q_name not in in_refs or not q_type or not reg.is_subtype(q_type, "table"):
                    continue
                tab_ep = max((epoch[n] for n in in_refs[q_name]), default=None)
                if val_ep is not None and tab_ep is not None and val_ep != tab_ep:
                    issues.append({"cell": cell.cell_id, "value_port": p_name, "value_vars": in_refs[p_name],
                                   "value_epoch": val_ep, "table_port": q_name, "table_vars": in_refs[q_name],
                                   "table_epoch": tab_ep})
        if out:
            ep = max((epoch[n] for refs in in_refs.values() for n in refs), default=0)
            if {str(e).lower() for e in (getattr(cell, "effects", None) or [])} & row_effects:
                ep += 1
            epoch[out] = ep
    return issues
