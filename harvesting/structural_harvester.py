"""
Tier 1 Deterministic Structural Harvester (No LLM, No Hallucination)

Extracts exact parameter signatures, required/optional flags, return types,
and minimal code templates directly from Python inspect/typing metadata
or OpenCV AST stubs (.pyi) and docstrings.
"""

import ast
import importlib
import inspect
import json
import os
import re
import site
import sys
import typing
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def clean_type_name(t_obj, default_type="Any") -> str:
    """Normalize raw python/typing type objects to clean type strings."""
    if t_obj is None or t_obj is inspect.Parameter.empty or t_obj is inspect.Signature.empty:
        return default_type

    if isinstance(t_obj, str):
        t_str = t_obj
    elif hasattr(t_obj, "__name__"):
        t_str = t_obj.__name__
    elif hasattr(t_obj, "_name") and t_obj._name:
        t_str = t_obj._name
    else:
        t_str = str(t_obj)

    # Strip module prefixes
    t_str = t_str.replace("typing.", "").replace("cv2.typing.", "").replace("pandas.core.frame.", "")
    t_str = t_str.replace("numpy.", "").replace("builtins.", "")

    # OpenCV Mat variants
    if "MatLike" in t_str or "UMat" in t_str or "Mat" in t_str:
        return "Mat"
    if "DataFrame" in t_str:
        return "DataFrame"
    if "Series" in t_str:
        return "Series"
    if "ndarray" in t_str:
        return "ndarray"

    # Union handling (e.g. Mat | None -> Mat)
    if "|" in t_str:
        parts = [p.strip() for p in t_str.split("|") if p.strip() not in ("None", "NoneType")]
        if parts:
            return clean_type_name(parts[0], default_type)

    if "[" in t_str:
        base = t_str.split("[")[0].strip()
        if base in ("Union", "Optional"):
            inner = t_str.split("[", 1)[1].rsplit("]", 1)[0]
            parts = [p.strip() for p in inner.split(",") if p.strip() not in ("None", "NoneType")]
            if parts:
                return clean_type_name(parts[0], default_type)
        return base

    return t_str.strip()


def harvest_cv2_structural() -> list[dict]:
    """Harvest structural nodes for OpenCV (cv2) using AST .pyi stubs + docstring fallbacks."""
    import cv2

    nodes = []
    seen_cells = set()

    # 1. Try parsing cv2/__init__.pyi from site-packages
    pyi_path = None
    for p in site.getsitepackages():
        candidate = os.path.join(p, "cv2", "__init__.pyi")
        if os.path.exists(candidate):
            pyi_path = candidate
            break

    if pyi_path and os.path.exists(pyi_path):
        try:
            with open(pyi_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for stmt in tree.body:
                if isinstance(stmt, ast.FunctionDef):
                    func_name = stmt.name
                    if func_name.startswith("_"):
                        continue

                    cell_id = f"CV2_{func_name.upper()}_DEFAULT"
                    if cell_id in seen_cells:
                        continue
                    seen_cells.add(cell_id)

                    args = stmt.args
                    num_pos = len(args.args)
                    num_defaults = len(args.defaults)
                    num_req = num_pos - num_defaults

                    params = []
                    req_placeholders = []
                    first_req_type = "Mat"

                    for i, arg in enumerate(args.args):
                        p_name = arg.arg
                        if p_name in ("self", "cls"):
                            continue
                        is_req = i < num_req
                        raw_ann = ast.unparse(arg.annotation) if arg.annotation else None
                        p_type = clean_type_name(raw_ann, "Mat" if p_name in ("src", "image", "img", "dst") else "int")
                        
                        if is_req and not req_placeholders:
                            first_req_type = p_type

                        default_val = None
                        if not is_req and i - num_req < len(args.defaults):
                            try:
                                default_val = ast.unparse(args.defaults[i - num_req])
                            except Exception:
                                default_val = None

                        params.append({
                            "name": p_name,
                            "type": p_type,
                            "required": is_req,
                            "default": default_val
                        })

                        if is_req:
                            req_placeholders.append(f"{{{p_name}}}")

                    ret_ann = ast.unparse(stmt.returns) if stmt.returns else None
                    out_type = clean_type_name(ret_ann, "Mat")

                    args_joined = ", ".join(req_placeholders) if req_placeholders else "{input_var}"
                    code = f"{{output_var}} = cv2.{func_name}({args_joined})"

                    nodes.append({
                        "cell_id": cell_id,
                        "domain": "opencv",
                        "name": func_name,
                        "params": params,
                        "inputs": [{"name": params[0]["name"] if params else "input_var", "type": first_req_type}],
                        "outputs": [{"name": "dst", "type": out_type}],
                        "domain_implementations": {
                            "Python_Core": {
                                "code": code,
                                "dependencies": ["cv2"]
                            }
                        },
                        "provenance": "structural",
                        "verified": False
                    })
        except Exception as e:
            print(f"[!] Warning: Failed to parse cv2 __init__.pyi: {e}")

    # 2. Walk cv2 module for any functions missing from .pyi stub
    for name, obj in inspect.getmembers(cv2):
        if name.startswith("_") or not (inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isroutine(obj)):
            continue

        cell_id = f"CV2_{name.upper()}_DEFAULT"
        if cell_id in seen_cells:
            continue
        seen_cells.add(cell_id)

        doc = getattr(obj, "__doc__", "") or ""
        first_line = doc.split("\n")[0].strip() if doc else ""
        
        req_args = []
        if "(" in first_line and ")" in first_line:
            args_str = first_line.split("(", 1)[1].split(")", 1)[0]
            req_part = args_str.split("[")[0]
            req_args = [a.strip() for a in req_part.split(",") if a.strip() and a.strip() != "self"]

        params = []
        req_placeholders = []
        for ra in req_args:
            p_type = "Mat" if ra in ("src", "img", "image", "dst") else "int"
            params.append({"name": ra, "type": p_type, "required": True, "default": None})
            req_placeholders.append(f"{{{ra}}}")

        if not params:
            params = [{"name": "src", "type": "Mat", "required": True, "default": None}]
            req_placeholders = ["{src}"]

        args_joined = ", ".join(req_placeholders)
        code = f"{{output_var}} = cv2.{name}({args_joined})"

        nodes.append({
            "cell_id": cell_id,
            "domain": "opencv",
            "name": name,
            "params": params,
            "inputs": [{"name": params[0]["name"], "type": params[0]["type"]}],
            "outputs": [{"name": "dst", "type": "Mat"}],
            "domain_implementations": {
                "Python_Core": {
                    "code": code,
                    "dependencies": ["cv2"]
                }
            },
            "provenance": "structural",
            "verified": False
        })

    return nodes


def harvest_python_library_structural(module_name: str) -> list[dict]:
    """Harvest structural nodes for a pure Python library (pandas, numpy, scipy, sklearn)."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        print(f"[!] Could not import module '{module_name}'")
        return []

    nodes = []
    visited = set()

    def process_func(obj, name, parent_name, is_method=False):
        dedup_key = f"{parent_name}.{name}"
        if dedup_key in visited:
            return
        visited.add(dedup_key)

        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            return

        try:
            hints = typing.get_type_hints(obj)
        except Exception:
            hints = {}

        params = []
        req_placeholders = []
        first_input_type = "DataFrame" if "pandas" in parent_name else "ndarray"

        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue

            raw_hint = hints.get(p_name)
            p_type = clean_type_name(raw_hint, "DataFrame" if "df" in p_name else ("ndarray" if "arr" in p_name or "a" == p_name or "input" in p_name or "X" in p_name or "y" in p_name else first_input_type))
            is_req = p.default is inspect.Parameter.empty
            default_val = None if is_req else repr(p.default)

            params.append({
                "name": p_name,
                "type": p_type,
                "required": is_req,
                "default": default_val
            })

            if is_req:
                req_placeholders.append(f"{{{p_name}}}")

        ret_hint = hints.get("return")
        out_type = clean_type_name(ret_hint, first_input_type)
        if out_type in ("Any", "any", "object", ""):
            out_type = first_input_type

        safe_name = f"{parent_name}_{name}".upper().replace(".", "_") + "_DEFAULT"

        if is_method:
            args_str = ", ".join(req_placeholders)
            code = f"{{output_var}} = {{input_var}}.{name}({args_str})"
        else:
            args_str = ", ".join(req_placeholders) if req_placeholders else "{input_var}"
            code = f"{{output_var}} = {parent_name}.{name}({args_str})"

        nodes.append({
            "cell_id": safe_name,
            "domain": module_name,
            "name": name,
            "params": params,
            "inputs": [{"name": params[0]["name"] if params else "input_var", "type": first_input_type}],
            "outputs": [{"name": "output", "type": out_type}],
            "domain_implementations": {
                "Python_Core": {
                    "code": code,
                    "dependencies": [module_name]
                }
            },
            "provenance": "structural",
            "verified": False
        })

    def walk_module(curr_mod, parent_path):
        if curr_mod in visited:
            return
        visited.add(curr_mod)

        for name, obj in inspect.getmembers(curr_mod):
            if name.startswith("_"):
                continue

            if inspect.ismodule(obj):
                mod_name = getattr(obj, "__name__", "")
                if mod_name.startswith(module_name) and obj not in visited:
                    walk_module(obj, mod_name)

            elif inspect.isclass(obj):
                if getattr(obj, "__module__", "").startswith(module_name):
                    if obj not in visited:
                        visited.add(obj)
                        for m_name, m_obj in inspect.getmembers(obj):
                            if m_name.startswith("_"):
                                continue
                            if inspect.isfunction(m_obj) or inspect.ismethod(m_obj) or inspect.isroutine(m_obj):
                                process_func(m_obj, m_name, f"{parent_path}.{name}", is_method=True)

            elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj):
                process_func(obj, name, parent_path, is_method=False)

    walk_module(mod, module_name)
    return nodes


def run_structural_harvest(library: str, output_file: str):
    print(f"[*] Running Tier 1 Structural Harvester for '{library}'...")
    if library == "cv2":
        nodes = harvest_cv2_structural()
    else:
        nodes = harvest_python_library_structural(library)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)

    print(f"[+] Harvested {len(nodes)} Tier 1 structural nodes into {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python structural_harvester.py <library_name> <output_file.json>")
        sys.exit(1)

    run_structural_harvest(sys.argv[1], sys.argv[2])
