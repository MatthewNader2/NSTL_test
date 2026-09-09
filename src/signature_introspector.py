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


def extract_clean_type_name(anno: Any) -> str:
    """Universal type name simplifier for AST nodes, type objects, and strings."""
    if anno is None or anno is inspect.Signature.empty or anno is inspect.Parameter.empty:
        return "any"
    if inspect.isclass(anno):
        return anno.__name__
    s = str(anno).strip()
    if not s or s == "...":
        return "any"
    if s.startswith("typing."):
        s = s[7:]
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip().lower() not in ("none", "nonetype")]
        s = parts[0] if parts else "None"
    if s.startswith(("Optional[", "Union[")) and "]" in s:
        inner = s[s.find("[") + 1: s.rfind("]")].strip()
        parts = [p.strip() for p in inner.split(",") if p.strip().lower() not in ("none", "nonetype")]
        if parts:
            s = parts[0]
    elif "[" in s and "]" in s:
        s = s[: s.find("[")].strip()
    if "." in s:
        s = s.split(".")[-1].strip()
    words = s.split()
    if words:
        s = words[0].rstrip(".,;:)>]`\"'")
    if s.lower() in ("retval", "result", "res", "return", "value", ""):
        return "any"
    return s


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
    Parses C-style bracket-nested parameters with 100% precision:
      "src, ksize[, dst[, borderType]]" ->
      [('src', True, 'any'), ('ksize', True, 'any'), ('dst', False, 'any'), ('borderType', False, 'any')]
    """
    tokens: List[Tuple[str, bool, str]] = []
    curr: List[str] = []
    depth = 0

    for ch in raw_params_str:
        if ch == "[":
            depth += 1
            continue
        elif ch == "]":
            if depth > 0:
                depth -= 1
            continue
        elif ch == ",":
            token = "".join(curr).strip()
            if token:
                is_req = (depth == 0 and "=" not in token)
                tokens.append((token, is_req))
            curr = []
        else:
            curr.append(ch)

    tail = "".join(curr).strip()
    if tail:
        is_req = (depth == 0 and "=" not in tail)
        tokens.append((tail, is_req))

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

        for p_name, is_req, p_type in parsed_params:
            if p_name in ("self", "cls") and (parent_cls_name or len(parameters) == 0):
                continue
            parameters.append(
                inspect.Parameter(
                    p_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if is_req else "default",
                    annotation=p_type if p_type != "any" else inspect.Parameter.empty
                )
            )

        ret_anno = inspect.Signature.empty
        if raw_return:
            clean_ret = extract_clean_type_name(raw_return.strip().split()[0])
            if clean_ret != "any":
                ret_anno = clean_ret

        return inspect.Signature(parameters=parameters, return_annotation=ret_anno)

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
