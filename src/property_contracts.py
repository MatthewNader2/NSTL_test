"""
src/property_contracts.py - Neuro-Symbolic Topological Lattice (NSTL)

Static propagation of TREE-DECLARED value properties.

Cells declare postconditions ("output_var.is_complex == True") and preconditions ("y.is_complex ==
False") in the trees, but until now those were only evaluated AFTER sandbox execution (the GEVR
contract), so a complex FFT result could be planned straight into an estimator that rejects it and
only fail at runtime ("Complex data not supported").

This module propagates known properties along the bound dataflow (variables, column projections,
column writes) and reports a consumer precondition that a KNOWN property contradicts. Unknown is never
a violation. Names of properties, columns, operations and libraries come only from the trees.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

import functools

Props = Dict[str, Tuple[Any, str, str]]   # property -> (value, producing cell id, description)


def _const(node: ast.AST) -> Optional[Any]:
    return node.value if isinstance(node, ast.Constant) else None


@functools.lru_cache(maxsize=1024)
def _parse_expr(expr_str: str) -> Optional[ast.AST]:
    try:
        return ast.parse(expr_str, mode="eval").body
    except SyntaxError:
        return None


def _known(expr: Any, props: Dict[str, Props], col_props: Dict[Tuple[str, Any], Props]) -> Props:
    tree = _parse_expr(str(expr))
    if tree is None:
        return {}
    return _eval(tree, props, col_props)


def _eval(node: ast.AST, props, col_props) -> Props:
    if isinstance(node, ast.Name):
        return dict(props.get(node.id, {}))
    if isinstance(node, ast.Call):
        # Function wrappers or single-argument calls like np.asarray(x), list(x)
        if len(node.args) == 1 and not node.keywords:
            arg_props = _eval(node.args[0], props, col_props)
            if arg_props:
                return arg_props
        # Attribute method calls like x.to_numpy(), x.copy()
        if isinstance(node.func, ast.Attribute):
            return _eval(node.func.value, props, col_props)
    if isinstance(node, ast.Attribute):
        # Attribute access like x.values
        return _eval(node.value, props, col_props)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        sl = node.slice
        cols: List[Any]
        if isinstance(sl, ast.List):
            cols = [_const(e) for e in sl.elts]
        else:
            cols = [_const(sl)]
        if not cols or any(c is None for c in cols):
            return {}
        per_col = [col_props.get((node.value.id, c), {}) for c in cols]
        out: Props = {}
        for prop in set().union(*[set(p) for p in per_col]):
            if all(prop in p for p in per_col) and len({repr(p[prop][0]) for p in per_col}) == 1:
                out[prop] = per_col[0][prop]
        return out
    return {}


def _cond_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Extracts condition field from either a dict or a ConditionPredicate/Pydantic object."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def analyze(pipeline_bindings: List[Tuple[Any, Dict[str, Any]]], orchestrator: Any = None) -> List[Dict[str, Any]]:
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
    except Exception:
        return []
    props: Dict[str, Props] = {}
    col_props: Dict[Tuple[str, Any], Props] = {}
    issues: List[Dict[str, Any]] = []

    for cell, bindings in pipeline_bindings:
        out = bindings.get("output_var")
        typed_bindings = getattr(cell, "typed_bindings", {}) or {}

        # 1. consumer preconditions against what is KNOWN
        for pre in (getattr(cell, "preconditions", None) or []):
            op = _cond_field(pre, "operator")
            if op not in ("==", "!="):
                continue
            port = _cond_field(pre, "target")
            prop = _cond_field(pre, "property")
            want = _cond_field(pre, "value")
            if port not in bindings or not prop:
                continue
            have = None
            rec = typed_bindings.get(port)
            if rec is not None:
                if rec.kind == "wire" and rec.root_var:
                    have = props.get(rec.root_var, {}).get(prop)
                elif rec.kind == "projection" and rec.root_var and rec.projection:
                    k = rec.projection.get("key")
                    if isinstance(k, (list, tuple, set)):
                        per_col = [col_props.get((rec.root_var, c), {}) for c in k]
                        have_props: Props = {}
                        for p in set().union(*[set(cp) for cp in per_col]):
                            if all(p in cp for cp in per_col) and len({repr(cp[p][0]) for cp in per_col}) == 1:
                                have_props[p] = per_col[0][p]
                        have = have_props.get(prop)
                    elif k is not None:
                        have = col_props.get((rec.root_var, k), {}).get(prop)
            if have is None:
                have = _known(bindings[port], props, col_props).get(prop)
            if have is None:
                continue
            satisfied = (have[0] == want) if op == "==" else (have[0] != want)
            if not satisfied:
                suggestions: List[str] = []
                if orchestrator is not None:
                    for c in orchestrator.loaded_cells.values():
                        if any(_cond_field(q, "property") == prop and _cond_field(q, "operator") == "=="
                               and _cond_field(q, "value") == want and _cond_field(q, "target") in ("output_var", None)
                               for q in (getattr(c, "postconditions", None) or [])):
                            suggestions.append(c.cell_id)
                issues.append({"cell": cell.cell_id, "port": port, "property": prop, "required": want,
                               "required_operator": op, "known": have[0], "bound": str(bindings[port]),
                               "established_by": have[1], "established_note": have[2],
                               "bridge_candidates": sorted(set(suggestions))[:6]})

        if not out:
            continue

        # 2. producer postconditions
        mine: Props = {}
        for post in (getattr(cell, "postconditions", None) or []):
            p_op = _cond_field(post, "operator")
            p_target = _cond_field(post, "target")
            p_prop = _cond_field(post, "property")
            if p_op == "==" and p_target in ("output_var", None) and p_prop:
                mine[p_prop] = (_cond_field(post, "value"), cell.cell_id, str(_cond_field(post, "description", "") or ""))
        # A cell may declare (tree effect "preserves:<property>") that its output keeps a property of its input
        # (e.g. a shift or reshape of a complex array is still complex). Without this, such cells silently
        # reset what is known and let a contradiction slip through.
        for eff in (getattr(cell, "effects", None) or []):
            if isinstance(eff, str) and eff.startswith("preserves:"):
                prop = eff.split(":", 1)[1].strip()
                for port_name in cell.inputs:
                    if port_name in bindings:
                        kn = None
                        rec = typed_bindings.get(port_name)
                        if rec is not None:
                            if rec.kind == "wire" and rec.root_var:
                                kn = props.get(rec.root_var, {})
                            elif rec.kind == "projection" and rec.root_var and rec.projection:
                                k = rec.projection.get("key")
                                if isinstance(k, str):
                                    kn = col_props.get((rec.root_var, k), {})
                        if kn is None:
                            kn = _known(bindings[port_name], props, col_props)
                        if prop in kn and prop not in mine:
                            mine[prop] = kn[prop]
                            break
        if mine:
            props[out] = {**props.get(out, {}), **mine}

        # 3. column writes: a cell that stores a value under a column key records the value's properties
        key_port = next((n for n, p in cell.inputs.items() if getattr(p, "binds", None) == "column_key"), None)
        val_port = next((n for n, p in cell.inputs.items() if getattr(p, "binds", None) == "assigned_value"), None)
        carrier_ports = [n for n, p in cell.inputs.items() if getattr(p, "binds", None) == "data_carrier"]
        for cp in carrier_ports:
            cp_rec = typed_bindings.get(cp)
            cp_var = cp_rec.root_var if (cp_rec and cp_rec.root_var) else str(bindings.get(cp))
            for (v, c), pr in list(col_props.items()):
                if v == cp_var:
                    col_props[(out, c)] = dict(pr)
        if key_port and val_port and key_port in bindings and val_port in bindings:
            k_rec = typed_bindings.get(key_port)
            col = str(k_rec.value_expr).strip().strip("'\"") if (k_rec and k_rec.value_expr) else str(bindings[key_port]).strip().strip("'\"")
            vp = None
            v_rec = typed_bindings.get(val_port)
            if v_rec and v_rec.root_var and v_rec.kind == "wire":
                vp = props.get(v_rec.root_var)
            if not vp:
                vp = _known(bindings[val_port], props, col_props)
            if vp:
                col_props[(out, col)] = vp
            else:
                col_props.pop((out, col), None)

        # 4. table-to-table cells keep what is known about their columns
        in_tables = []
        for n, p in cell.inputs.items():
            if n in bindings and reg.is_subtype(str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower(), "table"):
                rec_t = typed_bindings.get(n)
                in_tables.append(rec_t.root_var if (rec_t and rec_t.root_var) else str(bindings.get(n)))
        if in_tables and not (key_port and val_port):
            for (v, c), pr in list(col_props.items()):
                if v in in_tables and (out, c) not in col_props:
                    col_props[(out, c)] = dict(pr)
    return issues
