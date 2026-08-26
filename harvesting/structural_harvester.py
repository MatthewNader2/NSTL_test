"""
harvesting/structural_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Tier 1 Deterministic Structural Harvester: Extracts exact $k$-ary typed signatures,
parameter default values, and clean AST templates from Python AST, typing, and .pyi stubs.
"""

from __future__ import annotations
import ast
import importlib
import inspect
import json
import os
import site
import sys
import typing
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def canonical_type(raw_type_hint: Any, param_name: str = "", default: str = "any") -> str:
    """Normalizes raw Python type annotations to canonical lattice type names."""
    if raw_type_hint is None or raw_type_hint is inspect.Parameter.empty:
        p_l = param_name.lower()
        if p_l in ("path", "filename", "filepath", "uri", "source", "dest"):
            return "str"
        if p_l in ("df", "dataframe", "data"):
            return "DataFrame"
        if p_l in ("arr", "array", "matrix", "mat", "src", "img", "image"):
            return "ndarray"
        if p_l in ("s", "series"):
            return "Series"
        return default

    t_str = str(raw_type_hint)
    if hasattr(raw_type_hint, "__name__"):
        t_str = raw_type_hint.__name__

    t_str = t_str.replace("typing.", "").replace("cv2.typing.", "").replace("builtins.", "")
    t_str = t_str.replace("pandas.core.frame.", "").replace("numpy.", "")

    if "MatLike" in t_str or "Mat" in t_str or "UMat" in t_str:
        return "Mat"
    if "DataFrame" in t_str:
        return "DataFrame"
    if "Series" in t_str:
        return "Series"
    if "ndarray" in t_str:
        return "ndarray"
    if "str" in t_str:
        return "str"
    if "int" in t_str:
        return "int"
    if "float" in t_str:
        return "float"
    if "bool" in t_str:
        return "bool"
    if "dict" in t_str:
        return "dict"
    if "list" in t_str:
        return "list"

    # Union[T, None] / Optional[T] -> T
    if "|" in t_str:
        parts = [p.strip() for p in t_str.split("|") if p.strip() not in ("None", "NoneType")]
        if parts:
            return canonical_type(parts[0], param_name, default)

    if "[" in t_str:
        base = t_str.split("[")[0].strip()
        if base in ("Union", "Optional"):
            inner = t_str.split("[", 1)[1].rsplit("]", 1)[0]
            parts = [p.strip() for p in inner.split(",") if p.strip() not in ("None", "NoneType")]
            if parts:
                return canonical_type(parts[0], param_name, default)
        return base

    return t_str.strip() or default


def infer_typestate(param_name: str, type_name: str, is_output: bool = False) -> str:
    """Infers semantic typestate from parameter names and roles."""
    p_l = param_name.lower()
    if is_output:
        if type_name == "DataFrame":
            return "raw" if "read" in p_l else "computed"
        if type_name == "ndarray":
            return "computed"
        if type_name in ("bool", "None"):
            return "written" if "to_" in p_l or "write" in p_l else "computed"
        return "computed"

    if p_l in ("path", "filename", "filepath", "uri", "source"):
        return "source_identifier"
    if p_l in ("dest", "destination", "output_path", "out_filename"):
        return "dest_identifier"
    if type_name == "DataFrame":
        return "raw"
    return "any"


def harvest_cv2_structural() -> List[Dict[str, Any]]:
    """Harvests structural nodes for OpenCV from .pyi AST stubs and docstrings."""
    import cv2
    nodes = []
    seen = set()

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
                if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
                    func_name = stmt.name
                    cell_id = f"CV2_{func_name.upper()}_DEFAULT"
                    if cell_id in seen:
                        continue
                    seen.add(cell_id)

                    args = stmt.args
                    num_pos = len(args.args)
                    num_defaults = len(args.defaults)
                    num_req = num_pos - num_defaults

                    input_ports = {}
                    arg_placeholders = []

                    for i, arg in enumerate(args.args):
                        p_name = arg.arg
                        if p_name in ("self", "cls"):
                            continue
                        is_req = i < num_req
                        raw_ann = ast.unparse(arg.annotation) if arg.annotation else None
                        p_type = canonical_type(raw_ann, p_name, default="Mat" if p_name in ("src", "image", "img") else "int")
                        p_state = infer_typestate(p_name, p_type)

                        default_val = None
                        if not is_req and (i - num_req) < len(args.defaults):
                            try:
                                default_val = ast.unparse(args.defaults[i - num_req])
                            except Exception:
                                default_val = None

                        input_ports[p_name] = {
                            "type_name": p_type,
                            "state": p_state,
                            "required": is_req,
                            "default_value": default_val
                        }
                        if is_req:
                            arg_placeholders.append(f"{{{p_name}}}")

                    ret_ann = ast.unparse(stmt.returns) if stmt.returns else None
                    out_type = canonical_type(ret_ann, "dst", default="Mat")
                    out_state = infer_typestate(func_name, out_type, is_output=True)

                    args_str = ", ".join(arg_placeholders) if arg_placeholders else "{input_var}"
                    code = f"{{output_var}} = cv2.{func_name}({args_str})"

                    stage = 1 if func_name.startswith("imread") else (3 if func_name.startswith("imwrite") else 2)

                    nodes.append({
                        "cell_id": cell_id,
                        "domain_name": "opencv",
                        "node_type": "function",
                        "node_role": "function",
                        "stage": stage,
                        "keywords": [func_name.lower(), "cv2", "image"],
                        "inputs": input_ports,
                        "outputs": {"output_data": {"type_name": out_type, "state": out_state}},
                        "dependencies": ["import cv2"],
                        "code_template": code
                    })
        except Exception as e:
            print(f"[!] Warning: Failed to parse cv2 __init__.pyi: {e}")

    return nodes


def harvest_python_library_structural(module_name: str) -> List[Dict[str, Any]]:
    """Harvests structural nodes for Python modules (pandas, numpy, scipy, sklearn)."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        print(f"[!] Could not import '{module_name}'")
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

        input_ports = {}
        arg_placeholders = []
        default_in_type = "DataFrame" if "pandas" in module_name else "ndarray"

        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue

            raw_hint = hints.get(p_name)
            p_type = canonical_type(raw_hint, p_name, default=default_in_type)
            p_state = infer_typestate(p_name, p_type)
            is_req = p.default is inspect.Parameter.empty
            default_val = None if is_req else repr(p.default)

            input_ports[p_name] = {
                "type_name": p_type,
                "state": p_state,
                "required": is_req,
                "default_value": default_val
            }
            if is_req:
                arg_placeholders.append(f"{{{p_name}}}")

        ret_hint = hints.get("return")
        out_type = canonical_type(ret_hint, name, default=default_in_type)
        out_state = infer_typestate(name, out_type, is_output=True)

        cell_id = f"{parent_name}_{name}".upper().replace(".", "_") + "_DEFAULT"

        if is_method:
            args_str = ", ".join(arg_placeholders)
            code = f"{{output_var}} = {{input_var}}.{name}({args_str})"
        else:
            args_str = ", ".join(arg_placeholders) if arg_placeholders else "{input_var}"
            code = f"{{output_var}} = {parent_name}.{name}({args_str})"

        # Stage classification
        n_l = name.lower()
        if n_l.startswith("read_") or n_l.startswith("load_"):
            stage = 1
        elif n_l.startswith("to_") or n_l.startswith("save_") or n_l.startswith("export_"):
            stage = 3
        else:
            stage = 2

        import_alias = "import pandas as pd" if module_name == "pandas" else (
            "import numpy as np" if module_name == "numpy" else f"import {module_name}"
        )

        nodes.append({
            "cell_id": cell_id,
            "domain_name": module_name,
            "node_type": "function",
            "node_role": "function",
            "stage": stage,
            "keywords": [name.lower(), module_name, parent_name.split(".")[-1].lower()],
            "inputs": input_ports if input_ports else {"input_data": {"type_name": default_in_type, "state": "raw"}},
            "outputs": {"output_data": {"type_name": out_type, "state": out_state}},
            "dependencies": [import_alias],
            "code_template": code
        })

    def walk_module(curr_mod, parent_path):
        if curr_mod in visited:
            return
        visited.add(curr_mod)

        for name, obj in inspect.getmembers(curr_mod):
            if name.startswith("_"):
                continue

            if inspect.ismodule(obj) and getattr(obj, "__name__", "").startswith(module_name):
                walk_module(obj, obj.__name__)
            elif inspect.isclass(obj) and getattr(obj, "__module__", "").startswith(module_name):
                if obj not in visited:
                    visited.add(obj)
                    for m_name, m_obj in inspect.getmembers(obj):
                        if not m_name.startswith("_") and (inspect.isfunction(m_obj) or inspect.ismethod(m_obj)):
                            process_func(m_obj, m_name, f"{parent_path}.{name}", is_method=True)
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                process_func(obj, name, parent_path, is_method=False)

    walk_module(mod, module_name)
    return nodes


def run_structural_harvest(library: str, output_file: str):
    print(f"[*] Running Tier 1 Structural Harvester for '{library}'...")
    nodes = harvest_cv2_structural() if library == "cv2" else harvest_python_library_structural(library)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)

    print(f"[+] Harvested {len(nodes)} clean structural nodes into {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python structural_harvester.py <library_name> <output_file.json>")
        sys.exit(1)
    run_structural_harvest(sys.argv[1], sys.argv[2])
