"""
src/row_lineage.py - Universal Carrier Extent-Epoch & Key Def-Use Verifier.

Enforces two domain-agnostic dataflow invariants over any indexed/keyed carrier:
1. Extent-Epoch Alignment: Co-indexed inputs to a join/mutation cell must share the
   same extent epoch (incremented by any cell declaring extent/row/shape-mutating effects).
2. Causal Key Def-Use: A key/field introduced onto a carrier at step j cannot be
   projected from that carrier at an earlier step i < j.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Set, Tuple

try:
    from .lattice import TypeRegistry
    from .unification import resolve_typed_binding_record
except (ImportError, ValueError):
    from lattice import TypeRegistry
    from unification import resolve_typed_binding_record


def _referenced_vars(expr: Any, known: Dict[str, int]) -> List[str]:
    if not expr:
        return []
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError:
        return []
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in known]


def _extract_effect_names(cell: Any) -> Set[str]:
    if cell is None:
        return set()
    if hasattr(cell, "effect_names") and callable(cell.effect_names):
        return cell.effect_names()
    try:
        from .lattice import extract_effect_names
    except (ImportError, ValueError):
        from lattice import extract_effect_names
    return extract_effect_names(cell)


def analyze(pipeline_bindings: List[Tuple[Any, Dict[str, str]]]) -> List[Dict[str, Any]]:
    """
    Universal Extent-Epoch Alignment:
    Whenever a cell binds an 'assigned_value' (or secondary co-indexed payload) into a
    receiver/carrier, both inputs must originate from the same extent-mutation epoch.
    """
    try:
        reg = TypeRegistry.get_instance()
        mutating_effects = {
            str(e).lower()
            for key in ("extent_mutating_effects", "row_set_effects")
            for e in (reg.get_verification_semantics(key) or ())
        }
        carrier_roles = {
            str(r).lower()
            for r in (reg.get_verification_semantics("carrier_roles") or ("receiver", "carrier", "data_carrier"))
        }
    except Exception:
        return []

    if not mutating_effects:
        return []

    epoch: Dict[str, int] = {}
    issues: List[Dict[str, Any]] = []

    for cell, bindings in pipeline_bindings:
        if not isinstance(bindings, dict):
            continue
        out = bindings.get("output_var")
        in_refs: Dict[str, List[str]] = {
            p: _referenced_vars(v, epoch) for p, v in bindings.items() if p != "output_var"
        }
        inputs = getattr(cell, "inputs", {}) or {}

        for p_name, port in inputs.items():
            if getattr(port, "binds", None) != "assigned_value" or p_name not in in_refs:
                continue
            val_ep = max((epoch[n] for n in in_refs[p_name]), default=None)
            for q_name, q in inputs.items():
                if q_name == p_name or q_name not in in_refs:
                    continue
                q_role = str(getattr(q, "port_role", None) or getattr(q, "derived_role", "") or "").lower()
                q_binds = str(getattr(q, "binds", "") or "").lower()
                is_carrier = (
                    q_binds in carrier_roles
                    or q_role in carrier_roles
                    or q is getattr(cell, "primary_input", None)
                )
                if not is_carrier:
                    continue
                carrier_ep = max((epoch[n] for n in in_refs[q_name]), default=None)
                if val_ep is not None and carrier_ep is not None and val_ep != carrier_ep:
                    issues.append({
                        "cell": cell.cell_id,
                        "value_port": p_name,
                        "value_vars": in_refs[p_name],
                        "value_epoch": val_ep,
                        "carrier_port": q_name,
                        "carrier_vars": in_refs[q_name],
                        "carrier_epoch": carrier_ep,
                        "table_port": q_name,
                        "table_vars": in_refs[q_name],
                        "table_epoch": carrier_ep,
                    })

        if out:
            ep = max((epoch[n] for refs in in_refs.values() for n in refs), default=0)
            if _extract_effect_names(cell) & mutating_effects:
                ep += 1
            epoch[out] = ep

    return issues


def analyze_column_definitions(pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Universal Causal Key Def-Use:
    A key/field introduced onto a carrier at step j (via a key-binding port + assigned_value port)
    must not be projected from any carrier at an earlier step i < j.
    """
    origin_keys: Dict[str, Set[str]] = {}
    defined_at: Dict[str, int] = {}
    step_projected_keys: List[Tuple[int, Any, str, Set[str]]] = []

    key_bind_tags = {"key", "column_key", "field_key", "attr_key"}

    for idx, (cell, bindings) in enumerate(pipeline_bindings):
        if not isinstance(bindings, dict):
            continue
        inputs = getattr(cell, "inputs", {}) or {}
        typed_map = getattr(cell, "typed_bindings", {}) or {}

        def _keys_of_port(p_name: str, expr: Any) -> Set[str]:
            rec = typed_map.get(p_name) or resolve_typed_binding_record(cell, p_name, expr, inputs.get(p_name))
            found: Set[str] = set()
            if rec.kind == "projection" and rec.projection:
                k = rec.projection.get("key")
                if isinstance(k, (list, tuple, set)):
                    found.update(str(x).strip("'\"") for x in k)
                elif isinstance(k, str):
                    found.add(k.strip("'\""))
            elif rec.root_var and rec.root_var in origin_keys:
                found.update(origin_keys[rec.root_var])
            return found

        key_port = next((n for n, p in inputs.items() if getattr(p, "binds", None) in key_bind_tags and n in bindings), None)
        val_port = next((n for n, p in inputs.items() if getattr(p, "binds", None) == "assigned_value" and n in bindings), None)

        if key_port and val_port:
            written_key = str(bindings[key_port]).strip().strip("'\"")
            val_lineage = _keys_of_port(val_port, bindings[val_port])
            if written_key and written_key not in val_lineage:
                defined_at.setdefault(written_key, idx)

        acc: Set[str] = set()
        for p_name, expr in bindings.items():
            if p_name == "output_var" or p_name == key_port:
                continue
            p_keys = _keys_of_port(p_name, expr)
            if p_keys:
                acc |= p_keys
                step_projected_keys.append((idx, cell, p_name, p_keys))

        out = bindings.get("output_var")
        if out:
            origin_keys[out] = acc

    issues: List[Dict[str, Any]] = []
    for idx, cell, p_name, keys in step_projected_keys:
        for k in keys:
            if k in defined_at and defined_at[k] > idx:
                issues.append({
                    "cell": cell.cell_id,
                    "port": p_name,
                    "key": k,
                    "column": k,
                    "read_at": idx,
                    "defined_at": defined_at[k],
                    "defined_by": pipeline_bindings[defined_at[k]][0].cell_id,
                })
    return issues