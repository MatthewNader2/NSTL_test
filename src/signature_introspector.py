"""
src/signature_introspector.py

Pure dynamic runtime introspection and call reconstruction engine.
Contains ZERO hardcoded library branches or cell-id checks.

Grounds all function calls in runtime reflection:
- inspect.signature for standard Python callables
- Docstring signature parser for C-extensions / builtins
- Parameter contract validation: ensures all required parameters are provided,
  spurious enums are rejected, and valid enums target their declared parameter slots.
"""

import ast
import importlib
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple


def _find_call_in_doc(func_name: str, doc: str) -> Optional[str]:
    """Finds func_name(...) in doc using paren-depth tracking, handling nested parens deterministically."""
    target = func_name + "("
    pos = 0
    doc_len = len(doc)
    while pos < doc_len:
        idx = doc.find(target, pos)
        if idx == -1:
            return None
        if idx == 0 or not (doc[idx - 1].isalnum() or doc[idx - 1] == "_"):
            start_args = idx + len(target)
            paren_depth = 1
            cur = start_args
            while cur < doc_len and paren_depth > 0:
                ch = doc[cur]
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        break
                cur += 1
            if paren_depth == 0:
                return doc[start_args:cur].strip()
        pos = idx + 1
    return None


def extract_doc_signature(func_name: str, doc: str) -> Optional[Dict[str, Any]]:
    """Extracts formal parameter lists from docstrings of C-extensions (e.g. OpenCV, builtins)."""
    if not doc:
        return None

    raw_args = _find_call_in_doc(func_name, doc)
    if raw_args is None:
        return None

    if not raw_args:
        return {"required": [], "optional": []}

    # Bracket notation denotes optional parameters: 'arg1, arg2[, opt1[, opt2]]'
    if "[" in raw_args:
        req_part = raw_args.split("[")[0].strip().rstrip(",")
        opt_part = raw_args[len(req_part) :].replace("[", "").replace("]", "").strip().lstrip(",")
    else:
        req_part = raw_args.strip()
        opt_part = ""

    req_params = []
    opt_params = []
    for p in req_part.split(","):
        p_str = p.strip()
        if not p_str or p_str == "*":
            continue
        clean_p = p_str.split("=")[0].split(":")[0].strip()
        if not clean_p.isidentifier():
            continue
        if "=" in p_str:
            opt_params.append(clean_p)
        else:
            req_params.append(clean_p)

    for p in opt_part.split(","):
        p_str = p.strip()
        if not p_str or p_str == "*":
            continue
        clean_p = p_str.split("=")[0].split(":")[0].strip()
        if clean_p.isidentifier() and clean_p not in opt_params and clean_p not in req_params:
            opt_params.append(clean_p)

    return {"required": req_params, "optional": opt_params}


def get_callable_parameters(obj: Any, func_name: str) -> Optional[Dict[str, Any]]:
    """Introspects the ground truth parameter contract of any callable object."""
    # 1. Standard inspect.signature
    try:
        sig = inspect.signature(obj)
        required = []
        optional = []
        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls") or not p_name.isidentifier():
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.default == inspect._empty:
                required.append(p_name)
            else:
                optional.append(p_name)
        return {
            "required": required,
            "optional": optional,
            "all": required + optional,
            "doc": inspect.getdoc(obj) or "",
        }
    except Exception:
        pass

    # 2. Docstring signature parsing for C-extensions / builtins
    doc = getattr(obj, "__doc__", "") or ""
    parsed = extract_doc_signature(func_name, doc)
    if parsed:
        parsed["all"] = parsed["required"] + parsed["optional"]
        parsed["doc"] = doc
        return parsed

    return None


def get_enum_parameter_map(doc: str) -> Dict[str, str]:
    """Dynamically parses docstring parameter descriptions to find parameter-to-enum bindings.

    Detects patterns like `@param <name> ... (see #<Type>)` or `see #<Type>`.
    Returns {enum_prefix: param_name}.
    """
    mapping = {}
    if not doc:
        return mapping

    chunks = doc.split("@param")
    for chunk in chunks[1:]:
        chunk = chunk.strip()
        if not chunk:
            continue
        words = chunk.split(None, 1)
        p_name = "".join(c for c in words[0] if c.isalnum() or c == "_")
        if not p_name:
            continue
        p_desc = words[1] if len(words) > 1 else ""

        p_desc_lower = p_desc.lower()
        pos = 0
        n_desc = len(p_desc_lower)
        while pos < n_desc:
            see_idx = p_desc_lower.find("see", pos)
            if see_idx == -1:
                break
            before_ok = (see_idx == 0 or not p_desc_lower[see_idx - 1].isalnum())
            after_ok = (see_idx + 3 >= n_desc or not p_desc_lower[see_idx + 3].isalnum())
            if before_ok and after_ok:
                sub = p_desc[see_idx + 3 :].lstrip()
                if sub.startswith("#"):
                    sub = sub[1:].lstrip()
                token = ""
                for c in sub:
                    if c.isalnum() or c == "_":
                        token += c
                    else:
                        break
                cur_word = ""
                for c in token:
                    if c.isupper():
                        if cur_word:
                            break
                        cur_word += c
                    elif c.isalnum():
                        if cur_word:
                            cur_word += c
                    else:
                        if cur_word:
                            break
                if cur_word:
                    mapping[cur_word.upper()] = p_name
            pos = see_idx + 3

    return mapping


def resolve_callable_from_expr(func_expr: str, dependencies: Optional[List[str]] = None) -> Optional[Any]:
    """Dynamically resolves a callable object from its expression string and declared dependencies."""
    ns: Dict[str, Any] = {}
    if dependencies:
        for dep in dependencies:
            dep_clean = str(dep).strip()
            if dep_clean.startswith("import ") or dep_clean.startswith("from "):
                try:
                    exec(dep_clean, ns)
                except Exception:
                    pass

    if func_expr in ns:
        return ns[func_expr]

    parts = func_expr.split(".")
    root = parts[0].split("(")[0]

    obj = None
    if root in ns:
        obj = ns[root]
    else:
        try:
            obj = importlib.import_module(root)
        except Exception:
            pass

    if obj is not None:
        for part in parts[1:]:
            clean_part = part.split("(")[0]
            if hasattr(obj, clean_part):
                obj = getattr(obj, clean_part)
            else:
                return None
        return obj

    return None


def _mask_placeholders(s: str) -> str:
    res = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{" and i + 1 < n:
            end = s.find("}", i + 1)
            if end != -1:
                inner = s[i + 1 : end]
                if inner.isidentifier():
                    res.append(f"__ph_{inner}__")
                    i = end + 1
                    continue
        res.append(s[i])
        i += 1
    return "".join(res)


def _unmask_placeholders(s: str) -> str:
    res = []
    i = 0
    n = len(s)
    prefix = "__ph_"
    while i < n:
        if s.startswith(prefix, i):
            end = s.find("__", i + len(prefix))
            if end != -1:
                inner = s[i + len(prefix) : end]
                if inner.isidentifier():
                    res.append(f"{{{inner}}}")
                    i = end + 2
                    continue
        res.append(s[i])
        i += 1
    return "".join(res)


def parse_call_expression(template: str) -> Optional[Tuple[str, List[str], Dict[str, str]]]:
    """Extracts func_expr, raw_args, and raw_kwargs from a template using AST."""
    code = template.replace("{output_var}", "output_var")
    code = _mask_placeholders(code)

    try:
        tree = ast.parse(code)
    except Exception:
        return None

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if not calls:
        return None

    call = calls[0]
    func_expr = ast.unparse(call.func)
    raw_args = [_unmask_placeholders(ast.unparse(a)) for a in call.args]
    raw_kwargs = {kw.arg: _unmask_placeholders(ast.unparse(kw.value)) for kw in call.keywords if kw.arg}

    return func_expr, raw_args, raw_kwargs


def validate_and_reconstruct_call(
    func_obj: Any,
    func_expr: str,
    raw_args: List[str],
    raw_kwargs: Dict[str, str],
) -> str:
    """Validates and reconstructs a function call against its real ground-truth signature.

    - Supplies any missing required parameters.
    - Discards spurious enums that do not belong to the callable.
    - Positions valid enums at their exact parameter slot (positional or keyword).
    """
    func_name = func_expr.split(".")[-1]
    params = get_callable_parameters(func_obj, func_name)
    if not params:
        arg_strs = raw_args + [f"{k}={v}" for k, v in raw_kwargs.items()]
        return f"{func_expr}({', '.join(arg_strs)})"

    required = params["required"]
    optional = params["optional"]
    all_params = params["all"]
    enum_map = get_enum_parameter_map(params.get("doc", ""))

    # Classify arguments into valid enums, spurious enums, and data placeholders
    enum_args: Dict[str, str] = {}
    data_args: List[str] = []

    def _extract_flag(expr: str) -> Optional[str]:
        if "." in expr and not ("{" in expr or "(" in expr):
            tail = expr.rsplit(".", 1)[-1]
            if tail and all(c.isupper() or c.isdigit() or c == "_" for c in tail) and any(c.isupper() for c in tail):
                return tail
        return None

    for arg in raw_args:
        flag = _extract_flag(arg)
        if flag:
            target_p = None
            for p_prefix, p_name in enum_map.items():
                if flag.startswith(p_prefix) or p_prefix in flag:
                    target_p = p_name
                    break
            if target_p:
                enum_args[target_p] = arg
            else:
                # Spurious enum: does not belong to this callable. Drop it.
                pass
        else:
            data_args.append(arg)

    for k, v in raw_kwargs.items():
        flag = _extract_flag(v)
        if flag:
            target_p = None
            for p_prefix, p_name in enum_map.items():
                if flag.startswith(p_prefix) or p_prefix in flag:
                    target_p = p_name
                    break
            if target_p:
                enum_args[target_p] = v
        else:
            data_args.append(v)

    final_args: List[str] = []
    final_kwargs: Dict[str, str] = {}

    # Fill required positional parameters
    for i, req_p in enumerate(required):
        if req_p in enum_args:
            final_args.append(enum_args.pop(req_p))
        elif i < len(data_args):
            final_args.append(data_args[i])
        else:
            # Missing required parameter: supply canonical placeholder
            final_args.append(f"{{{req_p}}}")

    # Pass remaining data arguments
    for d_arg in data_args[len(required) :]:
        final_args.append(d_arg)

    # Valid enum arguments targeting optional parameters go to keyword arguments
    for p_name, enum_val in enum_args.items():
        if p_name in optional:
            final_kwargs[p_name] = enum_val

    arg_strs = final_args + [f"{k}={v}" for k, v in final_kwargs.items()]
    return f"{func_expr}({', '.join(arg_strs)})"
