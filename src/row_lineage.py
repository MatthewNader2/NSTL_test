"""
src/row_lineage.py - Universal Carrier Extent-Domain & Key-Space Verifier.

Enforces two domain-agnostic dataflow invariants over any indexed/keyed carrier:
1. Universal Extent-Domain Alignment:
   Every variable v in a pipeline has an extent_domain[v]: Tuple[str, int] = (root_carrier_var, extent_epoch).
   Any projection pi(v) inherits extent_domain[v].
   Any cell declaring an effect in extent_mutating_effects | row_set_effects increments the second element (root, epoch + 1).
   Any cell that combines a receiver carrier v_c and a co-indexed payload v_p (binds == "assigned_value" or co-indexed port)
   where extent_domain[v_c][0] == extent_domain[v_p][0] (same root carrier) but extent_domain[v_c][1] != extent_domain[v_p][1]
   (divergent extent epochs) emits an extent-misalignment issue.
2. Universal Key-Space Causal Def-Use:
   Every keyed carrier variable v tracks key_space[v] (added_keys, removed_keys, defined_at_step).
   Derived strictly from cell.typed_bindings (TypedBindingRecord) and port.binds / cell.effect_names(),
   with zero string slicing.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Set, Tuple, Optional

try:
    from .lattice import TypeRegistry
    from .unification import resolve_typed_binding_record
except (ImportError, ValueError):
    from lattice import TypeRegistry
    from unification import resolve_typed_binding_record


def _referenced_vars(expr: Any, known: Dict[str, Any]) -> List[str]:
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
    Universal Carrier Extent-Domain Alignment:
    Tracks extent_domain[v] = (root_carrier_var, extent_epoch) across the pipeline.
    Emits an issue whenever a receiver carrier v_c and co-indexed payload v_p
    share the same root carrier but differ in extent epoch.
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

    # extent_domain: var_name -> (root_carrier_var, extent_epoch)
    extent_domain: Dict[str, Tuple[str, int]] = {}
    issues: List[Dict[str, Any]] = []

    for cell, bindings in pipeline_bindings:
        if not isinstance(bindings, dict):
            continue
        inputs = getattr(cell, "inputs", {}) or {}
        typed_map = getattr(cell, "typed_bindings", {}) or {}

        def _resolve_extent(p_name: str, expr: Any) -> Optional[Tuple[str, int]]:
            rec = typed_map.get(p_name) or resolve_typed_binding_record(cell, p_name, expr, inputs.get(p_name))
            if rec and rec.root_var and rec.root_var in extent_domain:
                return extent_domain[rec.root_var]
            refs = _referenced_vars(expr, extent_domain)
            if refs:
                return extent_domain[refs[0]]
            return None

        # Check for co-indexed payload vs receiver carrier
        for p_name, port in inputs.items():
            if getattr(port, "binds", None) != "assigned_value" or p_name not in bindings:
                continue
            p_val = bindings[p_name]
            p_ext = _resolve_extent(p_name, p_val)
            if p_ext is None:
                continue

            for q_name, q in inputs.items():
                if q_name == p_name or q_name not in bindings:
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
                q_val = bindings[q_name]
                q_ext = _resolve_extent(q_name, q_val)
                if q_ext is None:
                    continue

                # Divergent extent epochs on co-indexed root carrier
                if p_ext[0] == q_ext[0] and p_ext[1] != q_ext[1]:
                    p_refs = _referenced_vars(p_val, extent_domain) or [str(p_val)]
                    q_refs = _referenced_vars(q_val, extent_domain) or [str(q_val)]
                    issues.append({
                        "cell": cell.cell_id,
                        "value_port": p_name,
                        "value_vars": p_refs,
                        "value_epoch": p_ext[1],
                        "carrier_port": q_name,
                        "carrier_vars": q_refs,
                        "carrier_epoch": q_ext[1],
                        "root_carrier": p_ext[0],
                        # Legacy compatibility keys
                        "table_port": q_name,
                        "table_vars": q_refs,
                        "table_epoch": q_ext[1],
                    })

        # Update extent domain for produced output variable
        out = bindings.get("output_var")
        if out:
            # Inherit root carrier from carrier input or first input
            in_exts = [_resolve_extent(p, v) for p, v in bindings.items() if p != "output_var"]
            valid_exts = [e for e in in_exts if e is not None]
            if valid_exts:
                root_var = valid_exts[0][0]
                ep = max(e[1] for e in valid_exts)
            else:
                root_var = out
                ep = 0

            if _extract_effect_names(cell) & mutating_effects:
                ep += 1
            extent_domain[out] = (root_var, ep)

    return issues


def analyze_column_definitions(pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Universal Key-Space Causal Def-Use:
    Tracks key_space[v] (added_keys, removed_keys, defined_at_step) derived strictly from
    cell.typed_bindings (TypedBindingRecord) and port.binds / cell.effect_names().
    A key projected from a carrier at step i must not be defined at an earlier step j > i.
    """
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
            if rec and rec.kind == "projection" and rec.projection:
                k = rec.projection.get("key")
                if isinstance(k, (list, tuple, set)):
                    found.update(str(x).strip("'\"") for x in k)
                elif isinstance(k, str):
                    found.add(k.strip("'\""))
            return found

        key_port = next((n for n, p in inputs.items() if getattr(p, "binds", None) in key_bind_tags and n in bindings), None)
        val_port = next((n for n, p in inputs.items() if getattr(p, "binds", None) == "assigned_value" and n in bindings), None)

        if key_port and val_port:
            written_key = str(bindings[key_port]).strip().strip("'\"")
            val_lineage = _keys_of_port(val_port, bindings[val_port])
            if written_key and written_key not in val_lineage:
                defined_at.setdefault(written_key, idx)

        for p_name, expr in bindings.items():
            if p_name == "output_var" or p_name == key_port:
                continue
            p_keys = _keys_of_port(p_name, expr)
            if p_keys:
                step_projected_keys.append((idx, cell, p_name, p_keys))

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