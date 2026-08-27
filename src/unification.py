"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate, Parameter Binding, and AST Code Synthesizer.
FIXED: robust filename quoting, expanded placeholder strategies, safer template binding.
"""

from __future__ import annotations
import ast
import os
import re
import sys
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from log_config import get_logger
from lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry

logger = get_logger('unification')


class UnificationFailure(Exception):
    pass


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

    def __init__(self, prompt: str = ""):
        self._scope: Dict[str, VariableBinding] = {}
        self.declared_dependencies: Set[str] = set()
        self._var_counter: Dict[str, int] = {}
        self._var_order: List[str] = []
        self.prompt_hint: str = prompt
        self.extracted_files: List[str] = []
        if prompt:
            self._extract_prompt_literals(prompt)

    def _extract_prompt_literals(self, prompt: str):
        # Match filenames with or without quotes. Captures WITHOUT quotes.
        ext_pattern = rf"\b([a-zA-Z0-9_\-./]+\.(?:{self.FILE_EXTENSIONS}))\b"
        files = re.findall(ext_pattern, prompt, re.IGNORECASE)
        files = list(dict.fromkeys(files))
        self.extracted_files = files

        if files:
            # ALWAYS quote filenames for safe Python insertion
            self.declare_variable(
                "input_file",
                AlgebraicSignature("str", "source_identifier"),
                literal_value=json.dumps(files[0])  # proper JSON quoting
            )
            if len(files) > 1:
                self.declare_variable(
                    "output_file",
                    AlgebraicSignature("str", "dest_identifier"),
                    literal_value=json.dumps(files[-1])
                )

        # Column extraction: sort by 'age', column 'name', etc.
        col_pattern = (
            r"(?:column|col|by|sort\s+by|group\s+by)\s+"
            r"(?:the\s+|a\s+|an\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?"
        )
        for match in re.finditer(col_pattern, prompt, re.IGNORECASE):
            col_name = match.group(1)
            if col_name.lower() not in self.COLUMN_STOP_WORDS:
                self.declare_variable(
                    "by_col",
                    AlgebraicSignature("str", "column_name"),
                    literal_value=json.dumps(col_name)
                )

        p_lower = prompt.lower()
        if re.search(r"\b(descending|desc|reverse|highest\s+to\s+lowest)\b", p_lower):
            self.declare_variable("sort_order", AlgebraicSignature("bool", "sort_flag"), literal_value="False")
        elif re.search(r"\b(ascending|asc|lowest\s+to\s+highest)\b", p_lower):
            self.declare_variable("sort_order", AlgebraicSignature("bool", "sort_flag"), literal_value="True")

    def declare_variable(
        self,
        base_name: str,
        signature: AlgebraicSignature,
        literal_value: Optional[str] = None,
        parent_var: Optional[str] = None
    ) -> str:
        clean_base = "".join(c if c.isalnum() or c == '_' else '_' for c in base_name).lower()
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"

        count = self._var_counter.get(clean_base, 0) + 1
        self._var_counter[clean_base] = count

        var_name = clean_base if count == 1 else f"{clean_base}_{count}"
        self._scope[var_name] = VariableBinding(
            name=var_name,
            signature=signature,
            literal_value=literal_value,
            lineage_parent=parent_var
        )
        self._var_order.append(var_name)
        return var_name

    def peek_next_variable_name(self, base_name: str) -> str:
        clean_base = "".join(c if c.isalnum() or c == '_' else '_' for c in base_name).lower()
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"
        count = self._var_counter.get(clean_base, 0) + 1
        return clean_base if count == 1 else f"{clean_base}_{count}"

    def find_compatible_variable(self, required_signature: AlgebraicSignature) -> Optional[VariableBinding]:
        for var_name in reversed(self._var_order):
            binding = self._scope[var_name]
            if binding.signature.unifies_with(required_signature):
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


class PlaceholderResolver:
    """
    Configurable, data-driven placeholder resolution engine.
    Resolution order:
      1. explicit_args
      2. ExecutionContext (prompt literals, scope variables)
      3. Cell configuration_schema defaults
      4. Fallback to None
    """

    KNOWN_FILE_EXTENSIONS = {
        'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'txt', 'db', 'h5', 'hdf5',
        'pdf', 'md', 'py', 'npz', 'pkl', 'pickle', 'feather', 'orc', 'avro', 'yaml', 'yml', 'toml', 'ini'
    }

    @classmethod
    def resolve(
        cls,
        placeholder: str,
        context: ExecutionContext,
        cell: Cell,
        explicit_args: Dict[str, Any],
        primary_input_override: Optional[str] = None
    ) -> str:
        ph = placeholder.strip()

        # 1. Explicit arguments take precedence
        if ph in explicit_args:
            val = explicit_args[ph]
            if isinstance(val, str):
                return json.dumps(val)
            return repr(val)

        # 2. PortSignature matching from cell.inputs
        port = cell.inputs.get(ph)
        if port is not None:
            sig = port.signature
            state = (sig.state or "").lower()
            type_name = (sig.type_name or "").lower()

            if state == "source_identifier":
                b = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
                if b and b.literal_value:
                    return b.literal_value
                if context.extracted_files:
                    return json.dumps(context.extracted_files[0])

            elif state == "dest_identifier":
                b = context.find_compatible_variable(AlgebraicSignature("str", "dest_identifier"))
                if b and b.literal_value:
                    return b.literal_value
                if context.extracted_files:
                    return json.dumps(context.extracted_files[-1])

            elif state == "column_name":
                b = context.find_compatible_variable(AlgebraicSignature("str", "column_name"))
                if b and b.literal_value:
                    return b.literal_value

            elif state == "sort_flag":
                b = context.find_compatible_variable(AlgebraicSignature("bool", "sort_flag"))
                if b and b.literal_value:
                    return b.literal_value

            elif type_name in ("dataframe", "mat", "ndarray", "dict", "list", "series"):
                if cell.stage != 1:
                    latest = context.get_latest_data_variable()
                    if latest:
                        return latest

            # Port default value
            if port.default_value is not None and port.default_value != "...":
                dv = str(port.default_value)
                if dv in ("True", "False", "None") or dv.replace('.', '', 1).isdigit():
                    return dv
                if any(dv.startswith(pfx) for pfx in ("cv2.", "np.", "pd.", "plt.", "scipy.", "sklearn.")):
                    return dv
                return json.dumps(dv)

        # 3. Semantic name fallback strategies
        ph_lower = ph.lower()
        if ph_lower in ("input_var", "df", "src", "img", "image", "data", "array", "mat", "x", "y"):
            if primary_input_override is not None:
                return primary_input_override
            latest = context.get_latest_data_variable()
            if latest:
                return latest

        if ph_lower in ("filepath", "filename", "input_filename", "source", "input_file", "path", "fname", "in_path"):
            b = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
            if b and b.literal_value:
                return b.literal_value
            if context.extracted_files:
                return json.dumps(context.extracted_files[0])

        if ph_lower in ("dest_path", "output_filename", "destination", "output_file", "dest", "out_path", "out_file"):
            b = context.find_compatible_variable(AlgebraicSignature("str", "dest_identifier"))
            if b and b.literal_value:
                return b.literal_value
            if context.extracted_files:
                return json.dumps(context.extracted_files[-1])

        if ph_lower in ("by", "by_column", "column", "columns", "cols", "axis", "index"):
            b = context.find_compatible_variable(AlgebraicSignature("str", "column_name"))
            if b and b.literal_value:
                return b.literal_value

        if ph_lower in ("ascending", "descending", "sort_flag"):
            b = context.find_compatible_variable(AlgebraicSignature("bool", "sort_flag"))
            if b and b.literal_value:
                return b.literal_value

        if ph_lower == "graph":
            b = context.find_compatible_variable(AlgebraicSignature("dict", "adjacency_dict"))
            if b and b.literal_value:
                return b.literal_value
            return "{'A': {'B': 1, 'C': 4}, 'B': {'A': 1, 'C': 2, 'D': 5}, 'C': {'A': 4, 'B': 2, 'D': 1}, 'D': {'B': 5, 'C': 1}}"

        if ph_lower == "start":
            b = context.find_compatible_variable(AlgebraicSignature("str", "source_node"))
            if b and b.literal_value:
                return b.literal_value
            return "'A'"

        # 4. Check configuration_schema defaults
        cfg = cell.configuration_schema or {}
        if isinstance(cfg, list):
            cfg = {p.get("name"): p for p in cfg if isinstance(p, dict) and "name" in p}

        if ph in cfg:
            param_meta = cfg[ph]
            if isinstance(param_meta, dict) and "default_value" in param_meta:
                dv = param_meta["default_value"]
                if dv is not None and dv != "...":
                    if isinstance(dv, str):
                        if any(dv.startswith(pfx) for pfx in ("cv2.", "np.", "pd.", "plt.", "scipy.", "sklearn.")):
                            return dv
                        return json.dumps(dv)
                    return repr(dv)
            if isinstance(param_meta, (str, int, float, bool)):
                return repr(param_meta)

        # 5. Check ExecutionContext variables by exact name
        binding = context.get_variable(ph)
        if binding and binding.literal_value is not None:
            return binding.literal_value
        if binding:
            return binding.name

        return "None"


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

    @staticmethod
    def unify_cell(
        context: ExecutionContext,
        cell: Cell,
        explicit_arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        explicit_args = explicit_arguments or {}

        # Stage-aware primary input resolution:
        if cell.stage == 1:
            file_binding = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
            if file_binding and file_binding.literal_value:
                primary_input_var = file_binding.literal_value
            elif context.extracted_files:
                primary_input_var = json.dumps(context.extracted_files[0])
            else:
                primary_input_var = "None"
        else:
            primary_input_var = context.get_latest_data_variable()
            if primary_input_var is None:
                primary_input_var = "None"

        cell_id_clean = cell.cell_id.lower().replace("_cell", "").replace("_default", "")
        raw_out_name = cell_id_clean.split('_')[-1]
        output_var_name = context.peek_next_variable_name(f"{raw_out_name}_out")

        if cell.dependencies:
            for dep in cell.dependencies:
                dep_str = str(dep).strip()
                if not dep_str:
                    continue
                mod = (
                    dep_str.replace("import", "").strip().split()[0].split(".")[0]
                    if dep_str.startswith("import") else dep_str.split(".")[0]
                )
                if mod.lower() in UnificationGate.DENYLISTED_IMPORTS:
                    continue
                context.declared_dependencies.add(dep_str)

        raw_code = cell.code_template
        if not raw_code:
            mod = UnificationGate._infer_module_prefix(cell)
            func = UnificationGate._infer_function_name(cell)
            arg = primary_input_var or "None"
            if mod:
                return f"{output_var_name} = {mod}.{func}({arg})"
            return f"{output_var_name} = {func}({arg})"

        transformed = raw_code.replace("{output_var}", output_var_name)

        placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", transformed))
        for ph in placeholders:
            resolved = PlaceholderResolver.resolve(
                ph, context, cell, explicit_args,
                primary_input_override=primary_input_var
            )
            transformed = transformed.replace(f"{{{ph}}}", resolved)

        # Safe unquoting: only strip quotes around known Python module constants (e.g. 'cv2.COLOR_BGR2GRAY')
        # NEVER strip quotes from filenames like 'data.csv'
        def _unquote_module_constant(m):
            const_expr = m.group(1)
            ext = const_expr.split('.')[-1].lower()
            if ext in PlaceholderResolver.KNOWN_FILE_EXTENSIONS:
                return m.group(0)  # Keep filename quoted!
            if any(const_expr.startswith(pfx) for pfx in ("cv2.", "np.", "pd.", "plt.", "scipy.", "sklearn.")):
                return const_expr
            return m.group(0)

        transformed = re.sub(
            r"['\"]([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            _unquote_module_constant,
            transformed
        )

        context.declare_variable(
            base_name=f"{raw_out_name}_out",
            signature=cell.primary_output,
            parent_var=primary_input_var if cell.stage != 1 else None
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
