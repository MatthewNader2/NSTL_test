"""
src/semantic_repair_engine.py

General dynamic semantic validation and repair engine for NSTL cells.
Contains ZERO hardcoded library checks or cell-name checks.

Given ANY cell in ANY tree:
1. Dynamically resolves the runtime callable invoked in `code_template`.
2. Validates the call against the ground truth signature contract (inspect /
   stubs / docstrings via signature_introspector.resolve_signature).
3. Discards spurious keyword arguments that do not belong to the callable.
4. Restores missing required positional parameters as template placeholders.
5. Synchronizes `inputs` with template placeholders (wiring invariant).

NOTE: this module previously imported three helpers
(`parse_call_expression`, `resolve_callable_from_expr`,
`validate_and_reconstruct_call`) that never existed anywhere in the
repository, so importing it always raised ImportError and the "Phase-2
semantic repair" feature was dead on arrival. This implementation wires the
promised behavior onto the real introspector API.
"""

from __future__ import annotations

import ast
import importlib
import re
from typing import Any, Dict, List, Optional, Tuple

from signature_introspector import resolve_signature
from template_wiring import (
    clean_malformed_template_braces,
    repair_wiring_invariant,
)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def parse_call_expression(template: str) -> Optional[Tuple[str, List[str], Dict[str, str]]]:
    """
    Parses the single call expression on the RHS of a code template.

    Template placeholders ({output_var}, {df}, ...) are substituted with
    syntactically valid identifiers before AST parsing, then the call's
    argument NAMES (positional placeholders / keyword names) are returned:
    (func_expr, positional_arg_names, keyword_arg_names).
    """
    if not template or "{" not in template:
        return None

    # Strip a leading assignment: "{output_var} = <rhs>" / "a, b = <rhs>"
    text = template.strip()
    m = re.match(r"^(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}(?:\s*,\s*\{[a-zA-Z_][a-zA-Z0-9_]*\})*)\s*=\s*(.+)$", text, re.DOTALL)
    rhs = m.group(1).strip() if m else text

    # Replace placeholders with valid identifiers so ast.parse succeeds.
    synthetic = {"_nstl_ph_{}".format(name) for name in _PLACEHOLDER_RE.findall(rhs)}
    ast_ready = _PLACEHOLDER_RE.sub(lambda mm: f"_nstl_ph_{mm.group(1)}", rhs)
    try:
        tree = ast.parse(ast_ready, mode="eval")
    except SyntaxError:
        return None

    call = tree.body
    while isinstance(call, ast.Attribute):
        call = call.value  # unwrap chained attributes to the base call
    if not isinstance(call, ast.Call):
        return None

    # Recover the dotted callee expression from the AST.
    def _unparse_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _unparse_name(node.value)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Call):
            return _unparse_name(node.func)
        return None

    func_expr = _unparse_name(call.func)
    if not func_expr:
        return None
    # Method calls on data carriers ({df}.head(...), {img}.resize(...)) have a
    # placeholder as their root: there is no static callable to resolve against,
    # so repair skips them gracefully.
    if func_expr.startswith("_nstl_ph_"):
        return None

    def _ph_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name) and node.id.startswith("_nstl_ph_"):
            return node.id[len("_nstl_ph_"):]
        return None

    positional: List[str] = []
    for arg in call.args:
        name = _ph_name(arg)
        if name is None:
            # A concrete (non-placeholder) argument: keep its source text.
            try:
                positional.append(ast.unparse(arg))
            except Exception:
                positional.append("")
        else:
            positional.append(name)

    keywords: Dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        name = _ph_name(kw.value)
        keywords[kw.arg] = name if name is not None else (ast.unparse(kw.value) if kw.value is not None else "")

    if not synthetic and not positional and not keywords:
        return None
    return func_expr, positional, keywords


def resolve_callable_from_expr(func_expr: str, dependencies: Optional[List[str]] = None):
    """
    Resolves a dotted expression (e.g. 'pd.read_csv', 'cv2.threshold') to the
    live callable using ONLY the cell's declared dependencies (alias -> module)
    plus the expression's own root module. Returns None when unresolvable.
    """
    if not func_expr:
        return None
    root, _, rest = func_expr.partition(".")

    candidate_modules: List[str] = []
    for dep in dependencies or []:
        dep_clean = str(dep).strip()
        if dep_clean.startswith(("import ", "from ")):
            dep_clean = dep_clean.replace("import ", "").replace("from ", "").split(" as ")[0].strip()
        base = dep_clean.split(".")[0].split(" ")[0]
        if not base:
            continue
        if base == root:
            # 'pandas as pd' style alias declared by this cell
            candidate_modules.append(dep_clean.replace(" as ", " ").split()[-1])
        else:
            candidate_modules.append(base)

    attr_chain = rest.split(".") if rest else []
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        obj = mod
        resolved_path = mod_name
        if not rest and root == mod_name.split(".")[0]:
            return mod
        ok = True
        for attr in attr_chain:
            nxt = getattr(obj, attr, None)
            if nxt is None:
                ok = False
                break
            obj = nxt
            resolved_path += f".{attr}"
        if ok and callable(obj):
            return obj
    # Last resort: the expression's own root module name.
    try:
        obj = importlib.import_module(root)
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj if callable(obj) else None
    except Exception:
        return None


def validate_and_reconstruct_call(
    func_obj: Any,
    func_expr: str,
    raw_args: List[str],
    raw_kwargs: Dict[str, str],
) -> str:
    """
    Validates a parsed call against the ground-truth signature of `func_obj`
    and reconstructs a corrected call expression:
      - keyword arguments not accepted by the callable are dropped,
      - required positional parameters missing from the call are restored as
        fresh placeholders ({param_name}),
      - positional placeholders are preserved in declared order.
    """
    sig = resolve_signature(func_obj)
    params: Dict[str, Any] = {}
    if isinstance(sig, dict):
        params = sig.get("parameters", {}) or {}
    else:
        params = sig or {}

    accepts_kwargs = bool(params.get("**kwargs")) if isinstance(params, dict) else False
    param_names = [p for p in (params.keys() if isinstance(params, dict) else []) if not str(p).startswith("*")]

    # 1. Drop spurious keyword arguments (not accepted by the callable).
    kept_kwargs: Dict[str, str] = {}
    for k, v in raw_kwargs.items():
        if accepts_kwargs or k in param_names or not param_names:
            kept_kwargs[k] = v

    # 2. Restore missing required positional parameters as placeholders.
    supplied = list(raw_args) + list(kept_kwargs.keys())
    restored: List[str] = []
    for p_name in param_names:
        if len(restored) + len(supplied) >= len(param_names) and supplied:
            break
        meta = params.get(p_name, {}) if isinstance(params, dict) else {}
        required = True
        if isinstance(meta, dict):
            required = bool(meta.get("required", True)) and meta.get("default_value") is None and meta.get("default") is None
        if p_name in supplied:
            continue
        if required:
            restored.append(p_name)
        else:
            break  # first optional gap ends positional restoration
    # 3. Preserve supplied positional order ahead of restored params.
    final_positional = list(raw_args)
    for r in restored:
        if r not in final_positional:
            final_positional.append(r)

    args_src = ", ".join(f"{{{a}}}" if _PLACEHOLDER_RE.fullmatch(a or "") else a for a in final_positional if a)
    kw_src = ", ".join(f"{k}={{{v}}}" if _PLACEHOLDER_RE.fullmatch(v or "") else f"{k}={v}" for k, v in kept_kwargs.items())
    call_src = f"{func_expr}({', '.join(x for x in (args_src, kw_src) if x)})"
    return call_src


def repair_cell_semantics(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Dynamically validates and repairs a single cell against ground-truth runtime signatures.

    Returns True if the cell was modified.
    """
    modified = False

    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    if tmpl != cell.get("code_template", ""):
        cell["code_template"] = tmpl
        modified = True

    call_info = parse_call_expression(tmpl)
    if call_info:
        func_expr, raw_args, raw_kwargs = call_info
        func_obj = resolve_callable_from_expr(func_expr, cell.get("dependencies"))

        if func_obj is not None and callable(func_obj):
            reconstructed = validate_and_reconstruct_call(func_obj, func_expr, raw_args, raw_kwargs)
            new_tmpl = f"{{output_var}} = {reconstructed}"

            if new_tmpl != tmpl:
                cell["code_template"] = clean_malformed_template_braces(new_tmpl)
                modified = True

    # Enforce wiring invariant (placeholder sets vs declared input ports)
    if repair_wiring_invariant(cell, domain):
        modified = True

    return modified
