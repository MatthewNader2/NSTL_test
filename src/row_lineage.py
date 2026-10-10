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
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
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


def analyze_column_definitions(pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    A column must not be READ before the cell that DEFINES it has run.

    A cell with a name-consuming port and a value-consuming port (tree: binds column_key / assigned_value)
    writes a column. The write DEFINES the column when the written value's own column lineage does not
    include it (Z written from an FFT of X); it UPDATES it when it does (X normalised and written back).
    Reading a defined-later column earlier in the pipeline is a guaranteed KeyError at runtime and used to
    pass lint (a model fitted on var_1['Z'] before any cell created Z).
    """
    origin: Dict[str, set] = {}

    def cols_of(node: ast.AST) -> set:
        if isinstance(node, ast.Name):
            return set(origin.get(node.id, set()))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.List) else [sl]
            return {e.value for e in elts if isinstance(e, ast.Constant)}
        out: set = set()
        for ch in ast.iter_child_nodes(node):
            out |= cols_of(ch)
        return out

    def parse(expr: Any):
        try:
            return ast.parse(str(expr), mode="eval").body
        except SyntaxError:
            return None

    def reads(node: ast.AST) -> set:
        found: set = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                sl = n.slice
                for e in (sl.elts if isinstance(sl, ast.List) else [sl]):
                    if isinstance(e, ast.Constant):
                        found.add(e.value)
        return found

    defined_at: Dict[Any, int] = {}
    for idx, (cell, b) in enumerate(pipeline_bindings):
        key = next((n for n, p in cell.inputs.items() if getattr(p, "binds", None) == "column_key" and n in b), None)
        val = next((n for n, p in cell.inputs.items() if getattr(p, "binds", None) == "assigned_value" and n in b), None)
        if key and val:
            col = str(b[key]).strip().strip("'\"")
            vnode = parse(b[val])
            if vnode is not None and col not in cols_of(vnode):
                defined_at.setdefault(col, idx)
        out = b.get("output_var")
        if out:
            acc: set = set()
            for port, expr in b.items():
                if port == "output_var":
                    continue
                node = parse(expr)
                if node is not None:
                    acc |= cols_of(node)
            origin[out] = acc

    issues: List[Dict[str, Any]] = []
    for idx, (cell, b) in enumerate(pipeline_bindings):
        for port, expr in b.items():
            if port == "output_var":
                continue
            node = parse(expr)
            if node is None:
                continue
            for col in reads(node):
                if col in defined_at and defined_at[col] > idx:
                    issues.append({"cell": cell.cell_id, "port": port, "column": col, "read_at": idx,
                                   "defined_at": defined_at[col], "defined_by": pipeline_bindings[defined_at[col]][0].cell_id})
    return issues
