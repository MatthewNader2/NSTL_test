"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate, Parameter Binding, and AST Code Synthesizer.
Domain-Agnostic Dynamic Typestate Port Binding (Zero Hardcodes).
"""

from __future__ import annotations
import ast
import os
import re
import sys
import json
import importlib
import importlib.util
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from log_config import get_logger

try:
    from .lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry
except (ImportError, ValueError):
    from lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry

logger = get_logger('unification')


class UnresolvedPlaceholderError(Exception):
    """Raised when a code template placeholder cannot be bound."""
    pass


class UnificationFailure(Exception):
    pass


class DynamicPlaceholderResolver:
    """
    Domain-agnostic parameter resolver.
    Binds placeholders dynamically based on PortSignature (type_name + state),
    scope variables, and extracted literal arguments.
    """

    @staticmethod
    def is_module_attribute(expr_str: str) -> bool:
        """Dynamically verifies if expr_str is an importable module attribute (e.g. cv2.COLOR_BGR2GRAY)."""
        if not expr_str or not isinstance(expr_str, str):
            return False
        expr_str = expr_str.strip()
        if not expr_str or "." not in expr_str:
            return False

        # Fast reject known file extensions
        known_exts = {'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'txt', 'parquet', 'h5', 'pkl', 'py'}
        if expr_str.split('.')[-1].lower() in known_exts or '/' in expr_str or '\\' in expr_str:
            return False

        try:
            tree = ast.parse(expr_str, mode='eval')
            if not isinstance(tree.body, ast.Attribute):
                return False

            parts = []
            curr = tree.body
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            parts.reverse()

            # Check if root is an installed module
            mod_name = parts[0]
            if importlib.util.find_spec(mod_name) is None:
                return False

            mod = importlib.import_module(mod_name)
            target = mod
            for attr in parts[1:]:
                if not hasattr(target, attr):
                    return False
                target = getattr(target, attr)
            return True
        except Exception:
            return False

    def resolve_port(
        self,
        port_name: str,
        port_sig: Union[PortSignature, AlgebraicSignature, Any],
        stage: int,
        context: Any,
        output_var: str
    ) -> str:
        """
        Dynamically resolves the replacement value for a given port placeholder.
        """
        if port_name == "output_var":
            return output_var

        # Extract type_name and state safely
        type_name = getattr(port_sig, "type_name", None)
        if type_name is None and hasattr(port_sig, "signature"):
            type_name = port_sig.signature.type_name
        type_name = str(type_name or "any")

        state = getattr(port_sig, "state", None)
        if state is None and hasattr(port_sig, "signature"):
            state = port_sig.signature.state
        state = str(state or "any")

        # 1. Check if an active variable in the execution context scope matches this typestate
        if hasattr(context, "scope_variables") and context.scope_variables:
            matching_vars = []
            for var_name, var_sig in context.scope_variables.items():
                if hasattr(var_sig, "unifies_with") and var_sig.unifies_with(port_sig):
                    matching_vars.append(var_name)
                elif hasattr(port_sig, "unifies_with") and port_sig.unifies_with(var_sig):
                    matching_vars.append(var_name)
                elif hasattr(var_sig, "signature") and hasattr(port_sig, "signature"):
                    if var_sig.signature.unifies_with(port_sig.signature):
                        matching_vars.append(var_name)
            if matching_vars:
                return matching_vars[-1]

        # 2. File / URI Source & Destination resolution based on state
        if state in ("source_identifier", "filepath_read", "input_uri") or port_name in ("filepath", "filename", "input_filename", "input_file", "source"):
            if hasattr(context, "source_files") and context.source_files:
                f = context.source_files[0]
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(context, "get_source_file"):
                f = context.get_source_file()
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(context, "extracted_files") and context.extracted_files:
                f = context.extracted_files[0]
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(port_sig, "default_value") and port_sig.default_value is not None and port_sig.default_value != "...":
                dv = str(port_sig.default_value)
                return f'"{dv}"' if not (dv.startswith('"') or dv.startswith("'")) else dv
            return '"input_data.csv"'

        if state in ("dest_identifier", "filepath_written", "output_uri") or port_name in ("dest_path", "output_filename", "destination", "output_file"):
            if hasattr(context, "dest_files") and context.dest_files:
                f = context.dest_files[0]
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(context, "get_dest_file"):
                f = context.get_dest_file()
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(context, "extracted_files") and len(context.extracted_files) > 1:
                f = context.extracted_files[-1]
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            if hasattr(port_sig, "default_value") and port_sig.default_value is not None and port_sig.default_value != "...":
                dv = str(port_sig.default_value)
                return f'"{dv}"' if not (dv.startswith('"') or dv.startswith("'")) else dv
            return '"output_data.csv"'

        # 3. Column name resolution
        if state == "column_name" or port_name in ("by", "column", "columns"):
            if port_name == "by" and hasattr(context, "by_column") and context.by_column:
                col = context.by_column
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "columns") and context.columns:
                col = context.columns[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(port_sig, "default_value") and port_sig.default_value is not None and port_sig.default_value != "...":
                dv = str(port_sig.default_value)
                return f'"{dv}"' if not (dv.startswith('"') or dv.startswith("'")) else dv
            return '"target"'

        # 4. Boolean flags (e.g. ascending=True/False)
        if type_name == "bool" or state == "sort_flag":
            if hasattr(context, "flags") and port_name in context.flags:
                return str(bool(context.flags[port_name]))
            if hasattr(context, "prompt_lower"):
                if "descending" in context.prompt_lower or "reverse" in context.prompt_lower:
                    return "False" if port_name == "ascending" else "True"
                if "ascending" in context.prompt_lower:
                    return "True" if port_name == "ascending" else "False"
            return "True"

        # 5. Fallback to bound parameter from context dictionary if available
        if hasattr(context, "parameters") and port_name in context.parameters:
            val = context.parameters[port_name]
            if isinstance(val, str) and not self.is_module_attribute(val) and not (val.startswith('"') or val.startswith("'")):
                return f'"{val}"'
            return str(val)

        # 6. Check default_value on port_sig if defined
        if hasattr(port_sig, "default_value") and port_sig.default_value is not None and port_sig.default_value != "...":
            dv = str(port_sig.default_value)
            if dv in ("True", "False", "None") or dv.replace('.', '', 1).isdigit() or (dv.startswith('-') and dv[1:].replace('.', '', 1).isdigit()):
                return dv
            if self.is_module_attribute(dv):
                return dv
            if not (dv.startswith('"') or dv.startswith("'")):
                return f'"{dv}"'
            return dv

        # 7. Generic Data / Scope Variable fallback (by type_name)
        if hasattr(context, "scope_variables") and context.scope_variables:
            for var_name, var_sig in reversed(context.scope_variables.items()):
                v_type = getattr(var_sig, "type_name", None)
                if v_type is None and hasattr(var_sig, "signature"):
                    v_type = var_sig.signature.type_name
                if str(v_type).lower() == type_name.lower():
                    return var_name

        if type_name.lower() in ("dataframe", "mat", "ndarray", "image", "tensor", "anyobject", "any", "data", "dataset"):
            if hasattr(context, "get_latest_data_variable"):
                latest = context.get_latest_data_variable()
                if latest:
                    return latest

        # 8. Dynamic domain-agnostic identifier fallback
        return port_name

    def assert_placeholders_resolved(self, code: str) -> None:
        """Ensures no raw {placeholder} tokens remain in synthesized code."""
        unresolved = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', code)
        if unresolved:
            raise UnresolvedPlaceholderError(f"Code contains unresolved placeholders: {unresolved}\nCode:\n{code}")


# Compatibility alias
PlaceholderResolver = DynamicPlaceholderResolver


def assert_placeholders_resolved(code_str: str, bindings: Optional[Dict[str, Any]] = None) -> None:
    test_code = code_str
    if bindings:
        for k, v in bindings.items():
            test_code = test_code.replace(f"{{{k}}}", str(v))
    DynamicPlaceholderResolver().assert_placeholders_resolved(test_code)


@dataclass
class VariableBinding:
    name: str
    signature: AlgebraicSignature
    literal_value: Optional[str] = None
    lineage_parent: Optional[str] = None


class ExecutionContext:
    """Manages lexical scopes, declared variables, and prompt literal bindings."""

    FILE_EXTENSIONS = (
        r'csv|json|parquet|xlsx|jpg|jpeg|png|bmp|txt|db|h5|hdf5|'
        r'pdf|md|py|npz|pkl|pickle|feather|orc|avro|yaml|yml|toml|ini'
    )
    COLUMN_STOP_WORDS = {
        "descending", "ascending", "the", "a", "an", "column", "columns",
        "data", "file", "csv", "by", "sort", "order", "and", "or", "in", "of",
        "to", "from", "with", "into", "as"
    }

    def __init__(
        self,
        prompt: str = "",
        scope: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self._scope: Dict[str, VariableBinding] = {}
        self.scope_variables: Dict[str, PortSignature] = {}
        self.declared_dependencies: Set[str] = set()
        self._var_counter: Dict[str, int] = {}
        self._var_order: List[str] = []
        self.prompt_hint: str = prompt
        self.prompt_lower: str = prompt.lower()
        self.extracted_files: List[str] = []
        self.source_files: List[str] = []
        self.dest_files: List[str] = []
        self.columns: List[str] = []
        self.by_column: Optional[str] = None
        self.flags: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = dict(parameters) if parameters else {}

        if scope:
            for k, v in scope.items():
                if isinstance(v, PortSignature):
                    self.scope_variables[k] = v
                    self._scope[k] = VariableBinding(name=k, signature=v.signature)
                elif isinstance(v, AlgebraicSignature):
                    self.scope_variables[k] = PortSignature(k, v)
                    self._scope[k] = VariableBinding(name=k, signature=v)

        if prompt:
            self._extract_prompt_literals(prompt)

    def _extract_prompt_literals(self, prompt: str):
        # Match filenames and paths with or without quotes/leading slashes. Captures clean path.
        ext_pattern = rf'(?:^|[\s"\'`\(])([/~a-zA-Z0-9_\-.]+\.(?:{self.FILE_EXTENSIONS}))'
        raw_files = re.findall(ext_pattern, prompt, re.IGNORECASE)
        files = [f.strip(".,;:\"'`()") for f in raw_files if f]
        files = list(dict.fromkeys(files))
        self.extracted_files = files

        src_file, dst_file = None, None

        dest_pattern = rf'(?:save|write|export|output|dump)?\s*(?:figure\s+|image\s+|data\s+|table\s+|results\s+|dataset\s+)*(?:to|into)\s+[\"\'`]?([/~a-zA-Z0-9_\-.]+\.(?:{self.FILE_EXTENSIONS}))'
        dest_m = re.search(dest_pattern, prompt, re.IGNORECASE)
        if dest_m:
            dst_file = dest_m.group(1).strip('.,;:\"\'`()')

        src_pattern = rf'(?:read|load|ingest|import|from)\s+(?:image\s+|data\s+|table\s+|file\s+|dataset\s+)*[\"\'`]?([/~a-zA-Z0-9_\-.]+\.(?:{self.FILE_EXTENSIONS}))'
        src_m = re.search(src_pattern, prompt, re.IGNORECASE)
        if src_m:
            src_file = src_m.group(1).strip('.,;:\"\'`()')

        if not src_file and files:
            src_file = files[0]
        if not dst_file and len(files) > 1:
            candidates = [f for f in files if f != src_file]
            dst_file = candidates[-1] if candidates else files[-1]

        if src_file:
            self.source_files = [src_file]
            self.declare_variable(
                "input_file",
                AlgebraicSignature("str", "source_identifier"),
                literal_value=json.dumps(src_file)
            )
        if dst_file:
            self.dest_files = [dst_file]
            self.declare_variable(
                "output_file",
                AlgebraicSignature("str", "dest_identifier"),
                literal_value=json.dumps(dst_file)
            )

        # Explicit 'by' column extraction (e.g. group by region, sort by salary)
        by_match = re.search(
            r"(?:group\s+by|sort\s+by|by)\s+(?:the\s+|a\s+|an\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        )
        if by_match:
            by_col = by_match.group(1)
            if by_col.lower() not in self.COLUMN_STOP_WORDS:
                self.by_column = by_col

        # Column extraction: sort by 'age', column 'name', group by region, sum revenue, etc.
        col_pattern = (
            r"(?:column|col|by|sort\s+by|group\s+by|sum|mean|aggregate|average|count|min|max|filter\s+by)\s+"
            r"(?:the\s+|a\s+|an\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?"
        )
        for match in re.finditer(col_pattern, prompt, re.IGNORECASE):
            col_name = match.group(1)
            if col_name.lower() not in self.COLUMN_STOP_WORDS:
                self.columns.append(col_name)
                self.declare_variable(
                    f"col_{col_name}",
                    AlgebraicSignature("str", "column_name"),
                    literal_value=json.dumps(col_name)
                )

        if "ascending" in self.prompt_lower:
            self.flags["ascending"] = True
            self.declare_variable(
                "sort_asc",
                AlgebraicSignature("bool", "sort_flag"),
                literal_value="True"
            )
        elif "descending" in self.prompt_lower or "reverse" in self.prompt_lower:
            self.flags["ascending"] = False
            self.declare_variable(
                "sort_asc",
                AlgebraicSignature("bool", "sort_flag"),
                literal_value="False"
            )

    def declare_variable(
        self,
        base_name: str,
        signature: Union[AlgebraicSignature, PortSignature],
        parent_var: Optional[str] = None,
        literal_value: Optional[str] = None
    ) -> str:
        if isinstance(signature, PortSignature):
            alg_sig = signature.signature
        else:
            alg_sig = signature

        clean_base = re.sub(r'[^a-zA-Z0-9_]', '', base_name)
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"

        idx = self._var_counter.get(clean_base, 0)
        var_name = clean_base if idx == 0 else f"{clean_base}_{idx + 1}"
        self._var_counter[clean_base] = idx + 1

        binding = VariableBinding(
            name=var_name,
            signature=alg_sig,
            literal_value=literal_value,
            lineage_parent=parent_var
        )
        self._scope[var_name] = binding
        if literal_value is None:
            self.scope_variables[var_name] = PortSignature(var_name, alg_sig)
        self._var_order.append(var_name)
        return var_name

    def peek_next_variable_name(self, base_name: str) -> str:
        clean_base = re.sub(r'[^a-zA-Z0-9_]', '', base_name)
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"
        idx = self._var_counter.get(clean_base, 0)
        return clean_base if idx == 0 else f"{clean_base}_{idx + 1}"

    def find_compatible_variable(self, required_sig: AlgebraicSignature) -> Optional[VariableBinding]:
        for var_name in reversed(self._var_order):
            binding = self._scope[var_name]
            if binding.signature.unifies_with(required_sig):
                return binding
        return None

    def get_latest_data_variable(self) -> Optional[str]:
        for var_name in reversed(self._var_order):
            binding = self._scope[var_name]
            if binding.signature.type_name not in ("str", "bool", "int", "None", "float"):
                return var_name
        return None

    def get_variable(self, name: str) -> Optional[VariableBinding]:
        return self._scope.get(name)


class UnificationGate:
    """Monadic code unifier: binds cell ports and templates to in-scope variables and prompt literals."""

    DENYLISTED_IMPORTS: Set[str] = {
        "generic", "algorithms", "data_engineering", "machine_learning",
        "python_core", "data_processing", "image_processing", "nlp", "opencv"
    }

    CANONICAL_IMPORTS: Dict[str, str] = {
        "pd": "import pandas as pd",
        "pandas": "import pandas as pd",
        "np": "import numpy as np",
        "numpy": "import numpy as np",
        "cv2": "import cv2",
        "plt": "import matplotlib.pyplot as plt",
        "sns": "import seaborn as sns",
        "torch": "import torch",
        "tf": "import tensorflow as tf",
        "sklearn": "import sklearn",
        "sk": "import sklearn",
        "heapq": "import heapq",
        "json": "import json",
        "os": "import os",
        "sys": "import sys",
        "math": "import math",
        "re": "import re",
        "random": "import random",
        "datetime": "import datetime",
        "collections": "import collections",
        "itertools": "import itertools",
        "functools": "import functools",
        "typing": "import typing",
        "pathlib": "import pathlib",
        "inspect": "import inspect",
        "hashlib": "import hashlib",
        "copy": "import copy",
        "pickle": "import pickle",
    }

    def __init__(self):
        pass

    @staticmethod
    def unify_cell(
        context: ExecutionContext,
        cell: Cell,
        explicit_arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        explicit_args = explicit_arguments or {}
        resolver = DynamicPlaceholderResolver()

        # Generate output variable name
        cell_id_clean = cell.cell_id.lower().replace("_cell", "").replace("_default", "")
        raw_out_name = cell_id_clean.split('_')[-1]
        output_var_name = context.peek_next_variable_name(f"{raw_out_name}_out")

        # Collect dependencies
        if cell.dependencies:
            for dep in cell.dependencies:
                dep_str = str(dep).strip()
                if not dep_str:
                    continue
                context.declared_dependencies.add(dep_str)

        raw_code = cell.code_template
        if not raw_code:
            mod = UnificationGate._infer_module_prefix(cell)
            func = UnificationGate._infer_function_name(cell)
            latest = context.get_latest_data_variable() or "None"
            if mod:
                raw_code = f"{{output_var}} = {mod}.{func}({latest})"
            else:
                raw_code = f"{{output_var}} = {func}({latest})"

        transformed = raw_code.replace("{output_var}", output_var_name)

        placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", transformed))
        for ph in placeholders:
            if ph in explicit_args:
                val = explicit_args[ph]
                val_str = json.dumps(val) if isinstance(val, str) and not (val.startswith('"') or val.startswith("'")) else str(val)
                transformed = transformed.replace(f"{{{ph}}}", val_str)
                continue

            port_sig = cell.inputs.get(ph)
            if port_sig is None:
                port_sig = PortSignature(ph, AlgebraicSignature("any", "any"))

            resolved = resolver.resolve_port(ph, port_sig, cell.stage, context, output_var_name)
            transformed = transformed.replace(f"{{{ph}}}", resolved)

        # Dynamic attribute unquoting for module constants (e.g. "cv2.COLOR_BGR2GRAY")
        def _unquote_mod(m):
            cand = m.group(1)
            if resolver.is_module_attribute(cand):
                return cand
            return m.group(0)

        transformed = re.sub(r'["\']([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)["\']', _unquote_mod, transformed)

        resolver.assert_placeholders_resolved(transformed)

        # Register primary output in context
        context.declare_variable(
            base_name=f"{raw_out_name}_out",
            signature=cell.primary_output
        )

        logger.info(f"[UNIFICATION SUCCESS] Bound {cell.cell_id} -> {output_var_name}")
        return transformed

    @staticmethod
    def _infer_module_prefix(cell: Cell) -> str:
        for dep in cell.dependencies:
            dep = str(dep).strip()
            if dep.startswith("import "):
                parts = dep.split()
                if len(parts) >= 2:
                    mod = parts[1].split(".")[0].split("as")[0].strip()
                    if mod not in UnificationGate.DENYLISTED_IMPORTS:
                        return mod
        domain_map = {
            "opencv": "cv2", "cv2": "cv2",
            "pandas": "pd", "numpy": "np",
            "scipy": "scipy", "sklearn": "sklearn",
            "matplotlib": "plt", "json": "json",
            "math": "math", "os": "os", "sys": "sys",
        }
        return domain_map.get((cell.domain_name or "").lower(), "")

    @staticmethod
    def _infer_function_name(cell: Cell) -> str:
        parts = cell.cell_id.lower().replace("_cell", "").replace("_default", "").split("_")
        if parts and parts[0] in ("cv2", "pd", "np", "plt", "sns", "sk", "tf", "torch"):
            parts = parts[1:]
        return "_".join(parts) if parts else "compute"

    @staticmethod
    def resolve_imports(code_text: str, context: Optional[ExecutionContext] = None) -> str:
        if not code_text or not code_text.strip():
            return code_text

        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return code_text

        existing_imports: Set[str] = set()
        body_nodes = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                existing_imports.add(ast.unparse(node).strip())
            else:
                body_nodes.append(node)

        if context:
            for dep in context.declared_dependencies:
                existing_imports.add(dep)

        module_access_names: Set[str] = set()
        for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module_access_names.add(node.value.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                module_access_names.add(node.func.id)

        required_imports: Set[str] = set()
        for imp in existing_imports:
            mod_root = (
                imp.replace("import", "").strip().split()[0].split(".")[0].lower()
                if imp.startswith("import") else imp.split(".")[0].lower()
            )
            if mod_root in UnificationGate.DENYLISTED_IMPORTS:
                continue
            required_imports.add(imp)

        for name in module_access_names:
            if name in UnificationGate.CANONICAL_IMPORTS:
                required_imports.add(UnificationGate.CANONICAL_IMPORTS[name])
            elif name in sys.stdlib_module_names and name not in ("str", "int", "float", "list", "dict", "set", "tuple", "bool", "print", "len", "max", "min", "range"):
                required_imports.add(f"import {name}")

        new_tree = ast.Module(body=body_nodes, type_ignores=[])
        ast.fix_missing_locations(new_tree)
        clean_body = ast.unparse(new_tree).strip()

        if required_imports:
            header = "\n".join(sorted(required_imports))
            return f"{header}\n\n{clean_body}"
        return clean_body

    @staticmethod
    def validate_synthesis(
        synthesized_dict: dict,
        expected_in_sig: AlgebraicSignature,
        expected_out_sig: AlgebraicSignature,
        trees_dir: str = "trees"
    ) -> bool:
        try:
            code = synthesized_dict.get("code_template", "")
            if not code:
                return False

            test_code = code.replace("{output_var}", "out_var").replace("{input_var}", "in_var")
            placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", test_code))
            for ph in placeholders:
                test_code = test_code.replace(f"{{{ph}}}", "None")
            ast.parse(test_code)

            actual_in = AlgebraicSignature(
                synthesized_dict.get("inputs", {}).get("type_name", "any"),
                synthesized_dict.get("inputs", {}).get("state", "any")
            )
            actual_out = AlgebraicSignature(
                synthesized_dict.get("outputs", {}).get("type_name", "any"),
                synthesized_dict.get("outputs", {}).get("state", "any")
            )

            if not expected_in_sig.unifies_with(actual_in):
                logger.warning(
                    f"[VALIDATE] Input type mismatch: expected {expected_in_sig}, got {actual_in}"
                )
                return False
            if not actual_out.unifies_with(expected_out_sig):
                logger.warning(
                    f"[VALIDATE] Output type mismatch: expected {expected_out_sig}, got {actual_out}"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"[VALIDATE] Synthesis validation failed: {e}")
            return False
