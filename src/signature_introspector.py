"""
src/signature_introspector.py - Neuro-Symbolic Topological Lattice (NSTL)
Multi-Tiered Universal Signature & Parameter Introspector.

Architecture:
  Tier 1: Native Reflection (inspect.signature with automatic unwrapping)
  Tier 2: Argument Clinic (__text_signature__ parsed via Python AST)
  Tier 3: PEP 561 Type Stubs (.pyi AST extraction for cv2, numpy, torch, etc.)
  Tier 4: Enhanced Docstring Parsing (handles ->, -->, C-style nested brackets [, ])
  Tier 5: Fail-Closed / Strictly Safe (skip when unknown; never guess 0 args)
"""

from __future__ import annotations
import ast
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Cache for parsed .pyi AST trees to ensure zero performance overhead
_STUB_CACHE: Dict[str, Optional[ast.Module]] = {}

# Clean regex for return arrows: supports '->', '-->', '--->'
_ARROW_RE = re.compile(r"\s*-+>\s*")

# Regex for callable signature header in docstrings
_DOC_SIG_RE = re.compile(
    r"^\s*(?:[a-zA-Z0-9_]+\.)*([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*-+>\s*(.*))?",
    re.MULTILINE
)


def _ast_clean_type(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            val = node.value.strip()
            while (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1].strip()
            if any(ch in val for ch in "[]|."):
                try:
                    inner_tree = ast.parse(val, mode="eval")
                    inner_res = _ast_clean_type(inner_tree.body)
                    if inner_res and inner_res != "any":
                        return inner_res
                except Exception:
                    pass
            return val
        return str(node.value)
    elif isinstance(node, ast.Subscript):
        val_name = _ast_clean_type(node.value)
        if val_name.lower() in ("union", "optional"):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                elts = [_ast_clean_type(e) for e in slice_node.elts if _ast_clean_type(e).lower() not in ("none", "nonetype")]
                return elts[0] if elts else "None"
            else:
                return _ast_clean_type(slice_node)
        elif val_name.lower() == "literal":
            return "str"
        return val_name
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _ast_clean_type(node.left)
        right = _ast_clean_type(node.right)
        if left.lower() in ("none", "nonetype"):
            return right
        return left
    elif isinstance(node, ast.Tuple):
        if node.elts:
            return _ast_clean_type(node.elts[0])
        return "tuple"
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ForwardRef":
        if node.args and isinstance(node.args[0], ast.Constant):
            return extract_clean_type_name(node.args[0].value)
    return "any"


def extract_clean_type_name(anno: Any) -> str:
    """Universal type name simplifier for AST nodes, type objects, and strings."""
    if anno is inspect.Signature.empty or anno is inspect.Parameter.empty:
        return "any"
    if anno is None or anno is type(None):
        return "None"
    if inspect.isclass(anno):
        return anno.__name__
    if hasattr(anno, "__name__"):
        return getattr(anno, "__name__")

    # Typing constructs with origin and ForwardRef
    try:
        import typing
        if isinstance(anno, getattr(typing, "ForwardRef", ())):
            return extract_clean_type_name(getattr(anno, "__forward_arg__", str(anno)))
        origin = typing.get_origin(anno)
        if origin is not None:
            if origin in (typing.Union, getattr(sys.modules.get("types", None), "UnionType", None)):
                args = [a for a in typing.get_args(anno) if getattr(a, "__name__", str(a)).lower() not in ("none", "nonetype")]
                if args:
                    return extract_clean_type_name(args[0])
                return "None"
            if hasattr(origin, "__name__"):
                return origin.__name__
    except Exception:
        pass

    s = str(anno).strip()
    while (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    if not s or s == "...":
        return "any"
    if s.startswith("typing."):
        s = s[7:]
    if s.startswith("ForwardRef(") and s.endswith(")"):
        s = s[len("ForwardRef("):-1].strip("\"'")

    # AST-level balanced parse for complex type strings
    try:
        tree = ast.parse(s, mode="eval")
        res = _ast_clean_type(tree.body)
        if res and res != "any":
            return res
    except Exception:
        pass

    # Safe fallback cleanup
    if s.startswith("ForwardRef(") and s.endswith(")"):
        s = s[len("ForwardRef("):-1].strip("\"'")
    if s.startswith("Optional[") and s.endswith("]"):
        return extract_clean_type_name(s[len("Optional["):-1].strip())
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip().lower() not in ("none", "nonetype")]
        s = parts[0] if parts else "None"
    if "[" in s:
        s = s[: s.find("[")].strip()
    if "." in s:
        s = s.split(".")[-1].strip()
    words = s.split()
    if words:
        s = words[0].strip(".,;:)>]`\"'([{<")
    if s.lower() in ("retval", "result", "res", "return", "value", ""):
        return "any"
    return s


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    parts = []
    depth = 0
    curr = []
    for ch in s:
        if ch in "([{<":
            depth += 1
            curr.append(ch)
        elif ch in ")]}>":
            depth = max(0, depth - 1)
            curr.append(ch)
        elif ch == sep and depth == 0:
            part = "".join(curr).strip()
            if part:
                parts.append(part)
            curr = []
        else:
            curr.append(ch)
    if curr:
        part = "".join(curr).strip()
        if part:
            parts.append(part)
    return parts


def infer_abstract_carrier(anno: Any, param_name: str = "") -> Optional[str]:
    """Infers abstract Python language-level carrier type without domain hardcodes."""
    if param_name:
        pl = param_name.lower()
        if pl in ("filepath", "filename", "file_path", "path", "file", "savepath", "pathname", "fname"):
            base_res = infer_abstract_carrier(anno) if (anno is not None and anno is not inspect.Signature.empty and anno is not inspect.Parameter.empty) else None
            if base_res in (None, "text", "any"):
                return "path"
            return base_res

    if anno is None or anno is inspect.Signature.empty or anno is inspect.Parameter.empty:
        return None

    import os
    import numbers
    import collections.abc

    # 1. Recursive typing unwrap (Optional, Union, Annotated, Literal, NewType)
    try:
        import typing
        origin = typing.get_origin(anno)
        if origin is not None:
            if origin in (typing.Union, getattr(sys.modules.get("types", None), "UnionType", None)):
                args = [a for a in typing.get_args(anno) if getattr(a, "__name__", str(a)).lower() not in ("none", "nonetype")]
                if len(args) == 1:
                    return infer_abstract_carrier(args[0])
                elif len(args) > 1:
                    resolved = [infer_abstract_carrier(a) for a in args]
                    valid = [r for r in resolved if r is not None]
                    if valid and len(set(valid)) == 1:
                        return valid[0]
                    if set(valid) == {"path", "text"} or ("path" in valid and all(v in ("path", "text") for v in valid)):
                        return "path"
            if origin in (list, dict, set, tuple, frozenset, collections.abc.Sequence, collections.abc.Mapping, collections.abc.Iterable):
                return "collection"
            if origin is getattr(typing, "Annotated", None):
                args = typing.get_args(anno)
                if args:
                    return infer_abstract_carrier(args[0])
            if origin is getattr(typing, "Literal", None):
                args = typing.get_args(anno)
                if args:
                    resolved = [infer_abstract_carrier(type(a)) for a in args]
                    valid = [r for r in resolved if r is not None]
                    if valid and len(set(valid)) == 1:
                        return valid[0]
        if hasattr(anno, "__supertype__"):
            return infer_abstract_carrier(anno.__supertype__)
    except Exception:
        pass

    # 2. Type-level protocol checks on classes
    if isinstance(anno, type) or inspect.isclass(anno):
        try:
            if issubclass(anno, os.PathLike) and (
                getattr(anno, "__module__", "") in ("os", "pathlib", "posixpath", "ntpath")
                or getattr(anno, "__name__", "").lower() in ("path", "filepath", "pathlike", "purepath", "posixpath", "windowspath")
            ):
                return "path"
            if issubclass(anno, bool):
                return "logical"
            if issubclass(anno, (numbers.Number, int, float, complex)):
                return "scalar"
            if issubclass(anno, (str, bytes, bytearray)):
                return "text"
            # Tabular / DataFrame protocol (__dataframe__)
            if hasattr(anno, "__dataframe__"):
                return "table"
            # Buffer / array protocol (PEP 688 collections.abc.Buffer, __array_interface__, __array__, __buffer__)
            if (
                hasattr(anno, "__array__")
                or hasattr(anno, "__array_interface__")
                or hasattr(anno, "__cuda_array_interface__")
                or hasattr(anno, "__buffer__")
            ):
                return "tensor"
            if hasattr(anno, "shape") and hasattr(anno, "dtype"):
                return "tensor"
            if hasattr(collections.abc, "Buffer") and issubclass(anno, getattr(collections.abc, "Buffer")):
                return "tensor"
            if issubclass(anno, (collections.abc.Sequence, collections.abc.Mapping, list, tuple, dict, set, frozenset)):
                return "collection"
        except Exception:
            pass

    # 3. String-based protocol and category classification
    name = str(getattr(anno, "__name__", anno)).strip()
    if not name or name.lower() in ("any", "object", "*", "unknown"):
        return None

    # Strip typing or collections.abc prefixes if present
    for prefix in ("typing.", "collections.abc.", "types.", "builtins."):
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Unwrap string pipe syntax (PEP 604)
    pipe_parts = _split_top_level(name, sep="|")
    if len(pipe_parts) > 1:
        parts = [p for p in pipe_parts if p.lower() not in ("none", "nonetype")]
        if len(parts) == 1:
            return infer_abstract_carrier(parts[0])
        elif len(parts) > 1:
            resolved = [infer_abstract_carrier(p) for p in parts]
            valid = [r for r in resolved if r is not None]
            if valid and len(set(valid)) == 1:
                return valid[0]
            if set(valid) == {"path", "text"} or ("path" in valid and all(v in ("path", "text") for v in valid)):
                return "path"

    # Unwrap string brackets Outer[Inner]
    if "[" in name and name.endswith("]"):
        bracket_idx = name.index("[")
        base = name[:bracket_idx].strip()
        inner = name[bracket_idx + 1 : -1].strip()
        base_lower = base.lower()
        if base_lower in ("optional",):
            return infer_abstract_carrier(inner)
        elif base_lower in ("annotated",):
            first_arg = _split_top_level(inner, sep=",")[0]
            return infer_abstract_carrier(first_arg)
        elif base_lower in ("literal",):
            args = _split_top_level(inner, sep=",")
            resolved = []
            for a in args:
                a_clean = a.strip()
                if (a_clean.startswith("'") and a_clean.endswith("'")) or (a_clean.startswith('"') and a_clean.endswith('"')):
                    resolved.append("text")
                elif a_clean in ("true", "false", "True", "False"):
                    resolved.append("logical")
                else:
                    try:
                        float(a_clean)
                        resolved.append("scalar")
                    except ValueError:
                        pass
            if resolved and len(set(resolved)) == 1:
                return resolved[0]
        elif base_lower in ("union",):
            union_parts = [p for p in _split_top_level(inner, sep=",") if p.lower() not in ("none", "nonetype")]
            if len(union_parts) == 1:
                return infer_abstract_carrier(union_parts[0])
            elif len(union_parts) > 1:
                resolved = [infer_abstract_carrier(p) for p in union_parts]
                valid = [r for r in resolved if r is not None]
                if valid and len(set(valid)) == 1:
                    return valid[0]
                if set(valid) == {"path", "text"} or ("path" in valid and all(v in ("path", "text") for v in valid)):
                    return "path"
        elif base_lower in (
            "sequence", "list", "tuple", "set", "frozenset", "dict",
            "mapping", "map", "iterable", "iterator", "generator",
            "collection", "queue", "deque"
        ):
            return "collection"
        else:
            res = infer_abstract_carrier(base)
            if res is not None:
                return res

    # Dynamically resolve string class name from sys.modules / builtins to query protocols
    if not isinstance(anno, type) and not inspect.isclass(anno):
        import builtins
        resolved_cls = getattr(builtins, name, None)
        if resolved_cls is None and "." in name:
            mod_part, cls_part = name.rsplit(".", 1)
            try:
                import importlib
                m_obj = importlib.import_module(mod_part)
                resolved_cls = getattr(m_obj, cls_part, None)
            except Exception:
                pass
            if resolved_cls is None:
                mod = sys.modules.get(mod_part)
                if mod:
                    resolved_cls = getattr(mod, cls_part, None)
        if resolved_cls is None:
            for mod in list(sys.modules.values()):
                if mod and hasattr(mod, name):
                    candidate = getattr(mod, name, None)
                    if inspect.isclass(candidate) or hasattr(candidate, "__origin__"):
                        resolved_cls = candidate
                        break
        if resolved_cls is not None and resolved_cls != anno:
            res = infer_abstract_carrier(resolved_cls)
            if res is not None:
                return res

    # Standard Python language primitives and ABC type names
    canonical = name.lower()
    if canonical in ("pathlike", "path", "filepath"):
        return "path"
    if canonical in ("int", "float", "complex", "number", "numeric"):
        return "scalar"
    if canonical in ("bool", "boolean", "logical"):
        return "logical"
    if canonical in ("str", "bytes", "bytearray", "text"):
        return "text"
    if (
        canonical in ("list", "tuple", "set", "frozenset", "dict", "mapping", "sequence", "iterable", "collection")
        or canonical.startswith(("sequence of", "list of", "tuple of", "dict of", "iterable of", "collection of"))
    ):
        return "collection"
    if canonical in ("buffer",) or canonical.startswith(("array-like", "array_like", "ndarray", "matrix", "tensor")):
        return "tensor"
    if canonical.startswith(("dataframe", "table")):
        return "table"

    return None


def extract_param_kind(param: inspect.Parameter) -> str:
    """Maps inspect.Parameter.kind to NSTL calling convention literal."""
    if param.kind is inspect.Parameter.POSITIONAL_ONLY:
        return "positional_only"
    elif param.kind is inspect.Parameter.KEYWORD_ONLY:
        return "keyword_only"
    elif param.kind is inspect.Parameter.VAR_POSITIONAL:
        return "var_positional"
    elif param.kind is inspect.Parameter.VAR_KEYWORD:
        return "var_keyword"
    return "standard"


def extract_docstring_params(doc: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses standard NumPy, Google, and Sphinx docstrings to extract documented parameter
    types and descriptions when runtime annotations are empty or generic.
    """
    params: Dict[str, Dict[str, Any]] = {}
    if not doc:
        return params

    # 1. NumPy style: Parameters\n----------\nx : int, optional\n    desc
    np_match = re.search(
        r"Parameters\s*\n\s*-+\s*\n(.*?)(?:\n\s*\n\s*[A-Z]|\n\s*Returns|\n\s*Raises|\n\s*Yields|\n\s*See Also|\Z)",
        doc,
        re.DOTALL,
    )
    if np_match:
        content = np_match.group(1)
        cur_name = None
        cur_type = ""
        cur_desc: List[str] = []
        base_indent = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(\s*)(\*{0,2}[a-zA-Z_]\w*)\s*:\s*(.*?)$", line)
            indent_len = len(m.group(1)) if m else 0
            if m and (base_indent is None or indent_len <= base_indent):
                if cur_name:
                    params[cur_name] = {"type": cur_type, "desc": " ".join(cur_desc)}
                cur_name = m.group(2).strip().lstrip("*")
                raw_type = m.group(3).strip()
                cur_type = re.sub(r",?\s*default\s*=.*$", "", raw_type, flags=re.IGNORECASE)
                cur_type = re.sub(r",?\s*optional\b.*$", "", cur_type, flags=re.IGNORECASE).strip()
                cur_desc = []
                base_indent = indent_len
            elif cur_name:
                cur_desc.append(stripped)
        if cur_name:
            params[cur_name] = {"type": cur_type, "desc": " ".join(cur_desc)}

    # 2. Google style: Args:\n    x (int): desc
    g_match = re.search(
        r"Args:\s*\n(.*?)(?:\n\s*\n\s*[A-Z]|\n\s*Returns|\n\s*Raises|\n\s*Yields|\Z)",
        doc,
        re.DOTALL,
    )
    if g_match:
        content = g_match.group(1)
        cur_name = None
        cur_type = ""
        cur_desc = []
        base_indent = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(\s*)([a-zA-Z_]\w*)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$", line)
            indent_len = len(m.group(1)) if m else 0
            if m and (base_indent is None or indent_len <= base_indent):
                if cur_name and cur_name not in params:
                    params[cur_name] = {"type": cur_type, "desc": " ".join(cur_desc)}
                cur_name = m.group(2).strip()
                cur_type = m.group(3).strip() if m.group(3) else ""
                first_desc = m.group(4).strip()
                cur_desc = [first_desc] if first_desc else []
                base_indent = indent_len
            elif cur_name:
                cur_desc.append(stripped)
        if cur_name and cur_name not in params:
            params[cur_name] = {"type": cur_type, "desc": " ".join(cur_desc)}

    # 3. Sphinx style: :param type name: desc  OR  :type name: type
    for m in re.finditer(r":param\s+(?:([a-zA-Z_]\w*)\s+)?([a-zA-Z_]\w*)\s*:\s*([^\n]+)", doc):
        t = m.group(1) or ""
        name = m.group(2)
        desc = m.group(3).strip()
        if name not in params:
            params[name] = {"type": t, "desc": desc}
        elif t and not params[name]["type"]:
            params[name]["type"] = t

    for m in re.finditer(r":type\s+([a-zA-Z_]\w*)\s*:\s*([^\n]+)", doc):
        name = m.group(1)
        t = m.group(2).strip()
        if name in params and not params[name]["type"]:
            params[name]["type"] = t
        elif name not in params:
            params[name] = {"type": t, "desc": ""}

    # 4. Doxygen / Javadoc style: @param name (optional type) desc
    for m in re.finditer(r"@param\s+([a-zA-Z_]\w*)\s*(?:\(([^)]+)\)\s*)?([^\n]+)", doc):
        name = m.group(1)
        t = m.group(2) or ""
        desc = m.group(3).strip()
        if name not in params:
            params[name] = {"type": t, "desc": desc}
        elif t and not params[name]["type"]:
            params[name]["type"] = t

    return params


def extract_param_constraints(text: str) -> Optional[Dict[str, Any]]:
    """Extracts numerical intervals, inequalities, and invariants from parameter docstrings."""
    if not text:
        return None
    constraints: Dict[str, Any] = {}

    # Intervals: [a, b], (a, b], [a, b), (a, b)
    m_int = re.search(r"([\[\(])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\]\)])", text)
    if m_int:
        left_bracket, min_val, max_val, right_bracket = m_int.groups()
        constraints["min"] = float(min_val) if "." in min_val else int(min_val)
        constraints["max"] = float(max_val) if "." in max_val else int(max_val)
        if left_bracket == "[" and right_bracket == "]":
            constraints["interval"] = "closed"
        elif left_bracket == "(" and right_bracket == "]":
            constraints["interval"] = "left_open"
        elif left_bracket == "[" and right_bracket == ")":
            constraints["interval"] = "right_open"
        else:
            constraints["interval"] = "open"

    # Inequalities
    if "min" not in constraints and "min_exclusive" not in constraints:
        m_gte = re.search(r">=\s*(-?\d+(?:\.\d+)?)", text)
        if m_gte:
            v = m_gte.group(1)
            constraints["min"] = float(v) if "." in v else int(v)
        elif re.search(r">\s*(-?\d+(?:\.\d+)?)", text):
            m_gt = re.search(r">\s*(-?\d+(?:\.\d+)?)", text)
            v = m_gt.group(1)
            constraints["min_exclusive"] = float(v) if "." in v else int(v)

    if "max" not in constraints and "max_exclusive" not in constraints:
        m_lte = re.search(r"<=\s*(-?\d+(?:\.\d+)?)", text)
        if m_lte:
            v = m_lte.group(1)
            constraints["max"] = float(v) if "." in v else int(v)
        elif re.search(r"<\s*(-?\d+(?:\.\d+)?)", text):
            m_lt = re.search(r"<\s*(-?\d+(?:\.\d+)?)", text)
            v = m_lt.group(1)
            constraints["max_exclusive"] = float(v) if "." in v else int(v)

    # Parity
    if re.search(r"\bodd\b", text, re.IGNORECASE):
        constraints["parity"] = "odd"
    elif re.search(r"\beven\b", text, re.IGNORECASE):
        constraints["parity"] = "even"

    # Positive / non-negative keywords
    if "positive" in text.lower() and "min" not in constraints and "min_exclusive" not in constraints:
        constraints["min_exclusive"] = 0
    if ("non-negative" in text.lower() or "nonnegative" in text.lower()) and "min" not in constraints:
        constraints["min"] = 0

    return constraints or None


def extract_shape_contract(text: str) -> Optional[Dict[str, Any]]:
    """Extracts expected rank / ndim tensor contracts from types or docstrings."""
    if not text:
        return None
    shape: Dict[str, Any] = {}
    m_ndim = re.search(r"\b([1-9]\d*)\s*-?\s*d(?:imensional)?\b", text, re.IGNORECASE)
    if m_ndim:
        shape["ndim"] = int(m_ndim.group(1))
    return shape or None


def extract_docstring_raises(doc: str) -> List[str]:
    """Extracts declared exception types from standard docstrings."""
    if not doc:
        return []
    raises: List[str] = []

    # 1. NumPy style: Raises\n------\nExceptionName\n    desc
    np_match = re.search(
        r"Raises\s*\n\s*-+\s*\n(.*?)(?:\n\s*\n\s*[A-Z]|\n\s*Returns|\n\s*Yields|\n\s*See Also|\Z)",
        doc,
        re.DOTALL,
    )
    if np_match:
        content = np_match.group(1)
        base_indent = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            indent_len = len(line) - len(line.lstrip(" "))
            if base_indent is None:
                base_indent = indent_len
            if indent_len <= base_indent:
                exc = stripped.split()[0].rstrip(":,")
                if exc.isidentifier() and exc[0].isupper():
                    raises.append(exc)

    # 2. Google style: Raises:\n    ExceptionName: desc
    g_match = re.search(
        r"Raises:\s*\n((?:\s+[A-Za-z_]\w*:[^\n]*\n*)+)",
        doc,
    )
    if g_match:
        for exc in re.findall(r"^\s+([A-Za-z_]\w*):", g_match.group(1), re.MULTILINE):
            if exc.isidentifier() and exc not in ("Args", "Raises", "Returns"):
                raises.append(exc)

    # 3. Sphinx style: :raises ExceptionName: desc
    for exc in re.findall(r":raises?\s+([A-Za-z_]\w*):", doc):
        if exc.isidentifier():
            raises.append(exc)

    return list(dict.fromkeys(raises))


def extract_ast_raises(source: Optional[str]) -> List[str]:
    """Extracts exception classes explicitly raised in callable AST source."""
    if not source:
        return []
    try:
        tree = ast.parse(source)
        raises: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc:
                if isinstance(node.exc, ast.Name):
                    raises.append(node.exc.id)
                elif isinstance(node.exc, ast.Call):
                    if isinstance(node.exc.func, ast.Name):
                        raises.append(node.exc.func.id)
                    elif isinstance(node.exc.func, ast.Attribute):
                        raises.append(node.exc.func.attr)
        return sorted(list(dict.fromkeys(raises)))
    except Exception:
        return []


def extract_type_vars(sig: inspect.Signature) -> List[str]:
    """Extracts generic type variables (T, K, V) declared across parameter and return annotations."""
    type_vars: Set[str] = set()
    import typing
    for p in sig.parameters.values():
        if isinstance(p.annotation, typing.TypeVar):
            type_vars.add(p.annotation.__name__)
        for arg in typing.get_args(p.annotation):
            if isinstance(arg, typing.TypeVar):
                type_vars.add(arg.__name__)
    if isinstance(sig.return_annotation, typing.TypeVar):
        type_vars.add(sig.return_annotation.__name__)
    for arg in typing.get_args(sig.return_annotation):
        if isinstance(arg, typing.TypeVar):
            type_vars.add(arg.__name__)
    return sorted(list(type_vars))


def extract_enum_domain(anno: Any, doc: str = "", param_name: str = "") -> Optional[List[Any]]:
    """Extracts allowable domain values (literals, enum members, or docstring choices)."""
    # 1. Inspect typing.Literal
    try:
        import typing
        if typing.get_origin(anno) is typing.Literal:
            args = list(typing.get_args(anno))
            if args:
                return args
    except Exception:
        pass

    # 2. Inspect enum.Enum class
    import enum
    if inspect.isclass(anno) and issubclass(anno, enum.Enum):
        try:
            return [m.name for m in anno]
        except Exception:
            pass

    # 3. Inspect string annotation for Literal[...]
    s = str(anno).strip()
    if "Literal[" in s:
        try:
            tree = ast.parse(s, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    val = node.value
                    if (isinstance(val, ast.Name) and val.id == "Literal") or (isinstance(val, ast.Attribute) and val.attr == "Literal"):
                        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                        vals = [ast.literal_eval(e) for e in elts if isinstance(e, ast.Constant)]
                        if vals:
                            return vals
        except Exception:
            pass

    # 4. Docstring inspection for choices: param_name : {a, b, c}
    if doc and param_name:
        pattern = rf"{re.escape(param_name)}\s*:[^{{\n]*\{{([^}}\n]+)\}}"
        m = re.search(pattern, doc, re.IGNORECASE)
        if m:
            raw_choices = m.group(1).split(",")
            cleaned = [c.strip().strip("'\"") for c in raw_choices if c.strip().strip("'\"")]
            if cleaned:
                return cleaned

    return None


def extract_docstring_returns(doc: str) -> List[Tuple[str, str, Optional[str]]]:
    """Extracts return specifications from standard docstrings (NumPy, Google, Sphinx)."""
    if not doc:
        return []
    # 1. NumPy / SciPy format
    m_sec = re.search(r"(?:Returns?|Yields?)\s*\n\s*[-=~]+\s*\n(.*?)(?:\n\s*\n\s*[A-Z][a-zA-Z0-9_ ]+|\Z)", doc, re.DOTALL)
    if m_sec:
        sec_text = m_sec.group(1)
        items = re.findall(r"^\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_.]+(?:\[[^\]]+\])?)", sec_text, re.MULTILINE)
        if items:
            valid = []
            for name, type_str in items:
                clean_t = extract_clean_type_name(type_str)
                if clean_t and clean_t.lower() not in ("none", "nonetype", "void"):
                    valid.append((name, clean_t, infer_abstract_carrier(clean_t)))
            if valid:
                return valid
        m_single = re.search(r"^\s*([a-zA-Z0-9_.]+(?:\[[^\]]+\])?)", sec_text, re.MULTILINE)
        if m_single:
            t = m_single.group(1).strip()
            clean_t = extract_clean_type_name(t)
            if clean_t and clean_t.lower() not in ("none", "nonetype", "void", "notes", "references", "see", "examples"):
                return [("output_data", clean_t, infer_abstract_carrier(clean_t))]

    # 2. Google docstring format
    m_google = re.search(r"(?:Returns?|Yields?):\s*\n\s*(?:(?:[a-zA-Z0-9_,\s]+)\s*:\s*)?([a-zA-Z0-9_.]+(?:\[[^\]]+\])?)", doc, re.MULTILINE)
    if m_google:
        t = m_google.group(1).strip()
        clean_t = extract_clean_type_name(t)
        if clean_t and clean_t.lower() not in ("none", "nonetype", "void"):
            return [("output_data", clean_t, infer_abstract_carrier(clean_t))]

    # 3. Sphinx / Epydoc format
    m_sphinx = re.search(r":(?:rtype|return|returns):\s*([a-zA-Z0-9_.]+(?:\[[^\]]+\])?)", doc, re.MULTILINE)
    if m_sphinx:
        t = m_sphinx.group(1).strip()
        clean_t = extract_clean_type_name(t)
        if clean_t and clean_t.lower() not in ("none", "nonetype", "void"):
            return [("output_data", clean_t, infer_abstract_carrier(clean_t))]

    return []


def extract_return_specs(ret_anno: Any, doc: str = "") -> List[Tuple[str, str, Optional[str]]]:
    """
    Extracts return port specifications: List[(port_name, type_name, abstract_type)].
    Handles both single returns and multi-return tuples / docstring unpacking.
    """
    first_line = doc.splitlines()[0] if doc else ""
    doc_out_names: Optional[List[str]] = None

    m_arrow = re.search(r"->\s*([a-zA-Z0-9_,\s]+)$", first_line)
    if m_arrow:
        parts = [p.strip() for p in m_arrow.group(1).split(",") if p.strip()]
        if len(parts) > 1 and all(p.isidentifier() for p in parts):
            doc_out_names = parts
    if not doc_out_names:
        m_assign = re.search(r"^\s*([a-zA-Z0-9_,\s]+)\s*=\s*[a-zA-Z0-9_.]+\(", first_line)
        if m_assign:
            parts = [p.strip() for p in m_assign.group(1).split(",") if p.strip()]
            if len(parts) > 1 and all(p.isidentifier() for p in parts):
                doc_out_names = parts

    tuple_element_types: List[str] = []
    try:
        import typing
        origin = typing.get_origin(ret_anno)
        if origin in (tuple, getattr(typing, "Tuple", None)):
            args = typing.get_args(ret_anno)
            if args and not (len(args) == 2 and args[1] is Ellipsis):
                tuple_element_types = [extract_clean_type_name(a) for a in args]
    except Exception:
        pass

    if not tuple_element_types and isinstance(ret_anno, str) and ("tuple[" in ret_anno.lower() or "Tuple[" in ret_anno):
        try:
            tree = ast.parse(ret_anno, mode="eval")
            if isinstance(tree.body, ast.Subscript):
                val = tree.body.value
                val_id = val.id if isinstance(val, ast.Name) else (val.attr if isinstance(val, ast.Attribute) else "")
                if val_id.lower() in ("tuple",):
                    slice_node = tree.body.slice
                    if isinstance(slice_node, ast.Tuple):
                        elts = [e for e in slice_node.elts if not (isinstance(e, ast.Constant) and e.value is Ellipsis)]
                        if len(elts) > 1:
                            tuple_element_types = [_ast_clean_type(e) for e in elts]
        except Exception:
            pass

    if tuple_element_types and len(tuple_element_types) > 1:
        results = []
        for idx, t in enumerate(tuple_element_types):
            port_name = doc_out_names[idx] if (doc_out_names and idx < len(doc_out_names)) else f"out_{idx}"
            results.append((port_name, t, infer_abstract_carrier(t)))
        return results

    if doc_out_names and len(doc_out_names) > 1:
        results = []
        for idx, name in enumerate(doc_out_names):
            results.append((name, "any", None))
        return results

    single_type = extract_clean_type_name(ret_anno)
    if single_type in ("any", ""):
        doc_specs = extract_docstring_returns(doc)
        if doc_specs:
            return doc_specs
    return [("output_data", single_type, infer_abstract_carrier(single_type))]


# =====================================================================
# TIER 1: Native Reflection & Unwrapping
# =====================================================================

def _tier1_inspect_signature(target: Any) -> Optional[inspect.Signature]:
    """Inspects native callable signatures, unwrapping decorators if present."""
    try:
        unwrapped = inspect.unwrap(target)
        return inspect.signature(unwrapped)
    except (ValueError, TypeError):
        return None
    except Exception:
        return None


# =====================================================================
# TIER 2: Argument Clinic (__text_signature__) via Python AST
# =====================================================================

def _tier2_text_signature(target: Any) -> Optional[inspect.Signature]:
    """
    Parses CPython Argument Clinic `__text_signature__` using Python's built-in AST.
    Eliminates regex guesswork for C builtins.
    """
    text_sig = getattr(target, "__text_signature__", None)
    if not text_sig or not isinstance(text_sig, str):
        return None

    cleaned = text_sig.strip()
    if not (cleaned.startswith("(") and cleaned.endswith(")")):
        return None

    # Sanitize Argument Clinic internal markers: ($module, $self, $type, etc.)
    params_body = cleaned[1:-1].strip()
    parts = [p.strip() for p in params_body.split(",") if p.strip()]
    sanitized_parts = []
    for p in parts:
        if p in ("$module", "$self", "$type"):
            continue
        if p.startswith(("$module", "$self", "$type")):
            continue
        sanitized_parts.append(p)

    dummy_code = f"def _dummy({', '.join(sanitized_parts)}): pass"
    try:
        tree = ast.parse(dummy_code)
    except SyntaxError:
        return None

    fn_def: ast.FunctionDef = tree.body[0]  # type: ignore
    args_node = fn_def.args

    # Determine default assignments
    num_defaults = len(args_node.defaults)
    pos_args = args_node.posonlyargs + args_node.args
    num_pos = len(pos_args)
    default_start_idx = num_pos - num_defaults

    parameters: List[inspect.Parameter] = []

    for idx, arg in enumerate(pos_args):
        p_name = arg.arg
        if idx >= default_start_idx:
            default_val: Any = "default"
        else:
            default_val = inspect.Parameter.empty

        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if idx < len(args_node.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        parameters.append(
            inspect.Parameter(p_name, kind=kind, default=default_val)
        )

    for idx, arg in enumerate(args_node.kwonlyargs):
        p_name = arg.arg
        kw_default_node = args_node.kw_defaults[idx]
        default_val = "default" if kw_default_node is not None else inspect.Parameter.empty
        parameters.append(
            inspect.Parameter(p_name, kind=inspect.Parameter.KEYWORD_ONLY, default=default_val)
        )

    return inspect.Signature(parameters=parameters)


# =====================================================================
# TIER 3: PEP 561 Type Stubs (.pyi) via AST Parsing
# =====================================================================

def _find_stub_file_for_module(mod: Any) -> Optional[Path]:
    """Locates the .pyi stub file corresponding to a loaded module or C-extension."""
    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        return None

    p = Path(mod_file)
    # Check directly matching stem (e.g., cv2.cpython-310-x86_64-linux-gnu.so -> cv2.pyi)
    base_stem = p.name.split(".")[0]

    candidates = [
        p.with_suffix(".pyi"),
        p.parent / f"{base_stem}.pyi",
        p.parent / "__init__.pyi",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def _get_stub_ast(stub_path: Path) -> Optional[ast.Module]:
    """Parses and caches the AST of a .pyi stub file."""
    path_str = str(stub_path)
    if path_str in _STUB_CACHE:
        return _STUB_CACHE[path_str]
    try:
        with open(stub_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code, filename=path_str)
        _STUB_CACHE[path_str] = tree
        return tree
    except Exception:
        _STUB_CACHE[path_str] = None
        return None


def _tier3_stub_signature(
    target: Any,
    callable_name: str,
    parent_cls_name: Optional[str] = None,
    mod: Optional[Any] = None
) -> Optional[inspect.Signature]:
    """
    Extracts 100% typed signatures from PEP 561 .pyi stub files.
    Eliminates heuristics for libraries like OpenCV (cv2), SciPy, NumPy, and PyTorch.
    """
    if mod is None:
        mod_name = getattr(target, "__module__", None)
        if mod_name and mod_name in sys.modules:
            mod = sys.modules[mod_name]
    if mod is None:
        return None

    stub_path = _find_stub_file_for_module(mod)
    if not stub_path:
        return None

    tree = _get_stub_ast(stub_path)
    if not tree:
        return None

    search_body: List[ast.AST] = tree.body
    if parent_cls_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == parent_cls_name:
                search_body = node.body
                break
        else:
            return None

    # Search for matching function definitions (including overloaded signatures)
    candidates: List[ast.FunctionDef] = []
    target_names = [callable_name]
    if parent_cls_name and callable_name.upper() == "INIT":
        target_names = ["__init__", "__new__"]

    for node in search_body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in target_names:
                candidates.append(node)  # type: ignore

    if not candidates:
        return None

    # In case of overloads, select the definition with the highest parameter coverage
    best_fn = max(candidates, key=lambda f: len(f.args.args) + len(f.args.kwonlyargs))

    parameters: List[inspect.Parameter] = []
    args_node = best_fn.args
    num_defaults = len(args_node.defaults)
    pos_args = args_node.posonlyargs + args_node.args
    num_pos = len(pos_args)
    default_start_idx = num_pos - num_defaults

    for idx, arg in enumerate(pos_args):
        p_name = arg.arg
        if p_name in ("self", "cls") and (parent_cls_name or idx == 0):
            continue

        p_anno = ast.unparse(arg.annotation) if arg.annotation else inspect.Parameter.empty
        if idx >= default_start_idx:
            default_val = "default"
        else:
            default_val = inspect.Parameter.empty

        parameters.append(
            inspect.Parameter(
                p_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_val,
                annotation=p_anno
            )
        )

    for idx, arg in enumerate(args_node.kwonlyargs):
        p_name = arg.arg
        p_anno = ast.unparse(arg.annotation) if arg.annotation else inspect.Parameter.empty
        kw_default_node = args_node.kw_defaults[idx]
        default_val = "default" if kw_default_node is not None else inspect.Parameter.empty
        parameters.append(
            inspect.Parameter(
                p_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default_val,
                annotation=p_anno
            )
        )

    ret_anno = ast.unparse(best_fn.returns) if best_fn.returns else inspect.Signature.empty
    return inspect.Signature(parameters=parameters, return_annotation=ret_anno)


# =====================================================================
# TIER 4: Enhanced Docstring Parsing (Arrow `-+>` & Bracket Unrolling)
# =====================================================================

def _unroll_bracketed_parameters(raw_params_str: str) -> List[Tuple[str, bool, str]]:
    """
    Parses C-style bracket-nested parameters and typed signatures with 100% precision.
    Distinguishes C-style optional brackets `[, opt]` from generic type brackets `arg: list[int]`.
    """
    tokens = []
    curr = []
    bracket_depth = 0
    type_depth = 0
    in_type = False
    tok_start_depth = 0

    for ch in raw_params_str:
        if ch == ":":
            in_type = True
            curr.append(ch)
            continue
        if in_type:
            if ch in "([<{":
                type_depth += 1
            elif ch in ")]>}":
                if type_depth > 0:
                    type_depth -= 1
                else:
                    in_type = False
            elif ch == "," and type_depth == 0:
                token = "".join(curr).strip()
                if token:
                    tokens.append((token, tok_start_depth == 0))
                curr = []
                in_type = False
                tok_start_depth = bracket_depth
                continue
            curr.append(ch)
            continue

        if ch == "[":
            bracket_depth += 1
            continue
        elif ch == "]":
            if bracket_depth > 0:
                bracket_depth -= 1
            continue
        elif ch == ",":
            token = "".join(curr).strip()
            if token:
                tokens.append((token, tok_start_depth == 0))
            curr = []
            tok_start_depth = bracket_depth
            continue
        else:
            if not curr and not ch.isspace():
                tok_start_depth = bracket_depth
            curr.append(ch)

    tail = "".join(curr).strip()
    if tail:
        tokens.append((tail, tok_start_depth == 0))

    results: List[Tuple[str, bool, str]] = []
    for raw_tok, is_req in tokens:
        tok = raw_tok.strip()
        if not tok or tok in ("/", "*"):
            continue
        if tok.startswith(("*", "**")):
            continue

        p_type = "any"
        p_name = tok
        if ":" in p_name:
            p_name, p_type_raw = p_name.split(":", 1)
            p_type = extract_clean_type_name(p_type_raw)
        if "=" in p_name:
            p_name = p_name.split("=")[0]
            is_req = False

        p_name = p_name.strip()
        if p_name.isidentifier():
            results.append((p_name, is_req, p_type))

    return results


def _tier4_docstring_signature(
    target: Any,
    callable_name: str,
    parent_cls_name: Optional[str] = None
) -> Optional[inspect.Signature]:
    """
    Enhanced fallback docstring parser that supports `-+>` (single, double, triple dashes)
    and unrolls nested C-extension optional brackets `[...]`.
    """
    doc = inspect.getdoc(target) or getattr(target, "__doc__", "") or ""
    if not doc:
        return None

    # Search for function signature pattern
    target_names = {callable_name.lower()}
    if parent_cls_name and callable_name.upper() == "INIT":
        target_names.add(parent_cls_name.lower())

    for match in _DOC_SIG_RE.finditer(doc):
        fn_name = match.group(1).lower()
        if fn_name not in target_names:
            continue

        raw_params = match.group(2).strip()
        raw_return = match.group(3)

        parsed_params = _unroll_bracketed_parameters(raw_params)
        parameters: List[inspect.Parameter] = []
        seen_default = False

        for p_name, is_req, p_type in parsed_params:
            if p_name in ("self", "cls") and (parent_cls_name or len(parameters) == 0):
                continue
            default_val = inspect.Parameter.empty if is_req else "default"
            if seen_default and default_val is inspect.Parameter.empty:
                default_val = "default"
            if default_val is not inspect.Parameter.empty:
                seen_default = True
            parameters.append(
                inspect.Parameter(
                    p_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_val,
                    annotation=p_type if p_type != "any" else inspect.Parameter.empty
                )
            )

        ret_anno = inspect.Signature.empty
        if raw_return:
            clean_ret = extract_clean_type_name(raw_return.strip().split()[0])
            if clean_ret != "any":
                ret_anno = clean_ret

        try:
            return inspect.Signature(parameters=parameters, return_annotation=ret_anno)
        except Exception:
            return None

    return None


# =====================================================================
# UNIFIED 5-TIER RESOLVER (Universal Entry Point)
# =====================================================================

def resolve_signature(
    target: Any,
    callable_name: str = "",
    parent_cls_name: Optional[str] = None,
    mod: Optional[Any] = None
) -> Optional[inspect.Signature]:
    """
    Resolves the signature through the full 5-tier architecture.
    Guarantees zero false assumptions: returns None if completely unknown.
    """
    # Tier 1: inspect.signature
    sig = _tier1_inspect_signature(target)
    if sig is not None:
        return sig

    # Tier 2: __text_signature__
    sig = _tier2_text_signature(target)
    if sig is not None:
        return sig

    # Tier 3: PEP 561 .pyi Stubs
    sig = _tier3_stub_signature(target, callable_name, parent_cls_name, mod)
    if sig is not None:
        return sig

    # Tier 4: Enhanced Docstring Parser
    sig = _tier4_docstring_signature(target, callable_name, parent_cls_name)
    if sig is not None:
        return sig

    # Tier 5: Fail-Closed (Unknown arity -> do not invent arguments)
    return None


def get_callable_parameters(callable_obj: Any, callable_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Maintains 100% backward compatibility with universal_harvester callers,
    upgraded to be powered by the complete 5-tier introspection engine.
    """
    sig = resolve_signature(callable_obj, callable_name)
    if sig is None:
        return None

    all_params: List[str] = []
    required_params: List[str] = []
    param_types: Dict[str, str] = {}

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        all_params.append(p.name)
        if p.default is inspect.Parameter.empty:
            required_params.append(p.name)
        param_types[p.name] = extract_clean_type_name(p.annotation)

    ret_type = extract_clean_type_name(sig.return_annotation)

    return {
        "all": all_params,
        "required": required_params,
        "types": param_types,
        "return_type": ret_type,
        "signature": sig
    }
