"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate, Parameter Binding, and AST Code Synthesizer.
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
        "data", "file", "csv", "by", "sort", "order", "and", "or", "in", "of"
    }

    def __init__(self, prompt: str = ""):
        self._scope: Dict[str, VariableBinding] = {}
        self.declared_dependencies: Set[str] = set()
        self._var_counter: Dict[str, int] = {}
        self._var_order: List[str] = []
        self.prompt_hint: str = prompt
        if prompt:
            self._extract_prompt_literals(prompt)

    def _extract_prompt_literals(self, prompt: str):
        ext_pattern = rf"[\'\"]?([a-zA-Z0-9_\-./]+\.(?:{self.FILE_EXTENSIONS}))[\'\"]?"
        files = re.findall(ext_pattern, prompt, re.IGNORECASE)
        files = list(dict.fromkeys(files))

        if files:
            self.declare_variable(
                "input_file",
                AlgebraicSignature("str", "source_identifier"),
                literal_value=repr(files[0])
            )
            if len(files) > 1:
                self.declare_variable(
                    "output_file",
                    AlgebraicSignature("str", "dest_identifier"),
                    literal_value=repr(files[-1])
                )

        col_pattern = (
            r"(?:column|col|by|sort\s+by)\s+"
            r"(?:the\s+|a\s+|an\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?"
        )
        for match in re.finditer(col_pattern, prompt, re.IGNORECASE):
            col_name = match.group(1)
            if col_name.lower() not in self.COLUMN_STOP_WORDS:
                self.declare_variable(
                    "by_col",
                    AlgebraicSignature("str", "column_name"),
                    literal_value=repr(col_name)
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
            if binding.signature.type_name not in ("str", "bool", "int", "None"):
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

    DEFAULT_STRATEGIES: Dict[str, str] = {
        "input_var": "primary_data",
        "df": "primary_data",
        "src": "primary_data",
        "img": "primary_data",
        "image": "primary_data",
        "data": "primary_data",
        "graph": "primary_data",
        "array": "primary_data",
        "mat": "primary_data",
        "filepath": "source_file",
        "filename": "source_file",
        "input_filename": "source_file",
        "source": "source_file",
        "input_file": "source_file",
        "dest_path": "dest_file",
        "output_filename": "dest_file",
        "destination": "dest_file",
        "output_file": "dest_file",
        "dest": "dest_file",
        "by": "column_name",
        "by_column": "column_name",
        "ascending": "sort_flag",
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

        if ph in explicit_args:
            return repr(explicit_args[ph])

        strategy = cls.DEFAULT_STRATEGIES.get(ph, "unknown")
        if strategy == "primary_data" and primary_input_override is not None:
            context_result = primary_input_override
        else:
            resolver = getattr(cls, f"_resolve_{strategy}", cls._resolve_unknown)
            context_result = resolver(context, cell, ph)

        if context_result is not None:
            return context_result

        # Schema defaults (only if context has no binding)
        cfg = cell.configuration_schema or {}
        if isinstance(cfg, list):
            # Convert list-format params to dict for lookup
            cfg = {p.get("name"): p for p in cfg if isinstance(p, dict) and "name" in p}

        if ph in cfg:
            param_meta = cfg[ph]
            if isinstance(param_meta, dict) and "default_value" in param_meta:
                return str(param_meta["default_value"])
            if isinstance(param_meta, (str, int, float, bool)):
                return repr(param_meta)

        return "None"

    @classmethod
    def _resolve_primary_data(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        return context.get_latest_data_variable()

    @classmethod
    def _resolve_source_file(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        b = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
        return b.literal_value if (b and b.literal_value) else None

    @classmethod
    def _resolve_dest_file(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        b = context.find_compatible_variable(AlgebraicSignature("str", "dest_identifier"))
        return b.literal_value if (b and b.literal_value) else None

    @classmethod
    def _resolve_column_name(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        b = context.find_compatible_variable(AlgebraicSignature("str", "column_name"))
        if b and b.literal_value:
            return b.literal_value
        return None

    @classmethod
    def _resolve_sort_flag(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        b = context.find_compatible_variable(AlgebraicSignature("bool", "sort_flag"))
        return b.literal_value if (b and b.literal_value) else None

    @classmethod
    def _resolve_unknown(cls, context: ExecutionContext, cell: Cell, ph: str) -> Optional[str]:
        binding = context.get_variable(ph)
        if binding and binding.literal_value is not None:
            return binding.literal_value
        if binding:
            return binding.name
        return None


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

        # Stage-aware input resolution:
        # Stage 1 (Source) cells ingest file paths / initial parameters.
        # Stage 2+ cells consume data variables from the pipeline.
        if cell.stage == 1:
            file_binding = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
            if file_binding and file_binding.literal_value:
                primary_input_var = file_binding.literal_value
            else:
                # Fallback: any string literal in context (e.g., extracted filenames)
                primary_input_var = None
                for name in reversed(context._var_order):
                    binding = context._scope[name]
                    if binding.signature.type_name == "str" and binding.literal_value:
                        primary_input_var = binding.literal_value
                        break
                if primary_input_var is None:
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

        placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", transformed))
        for ph in placeholders:
            resolved = PlaceholderResolver.resolve(
                ph, context, cell, explicit_args,
                primary_input_override=primary_input_var
            )
            transformed = transformed.replace(f"{{{ph}}}", resolved)

        # Post-process: strip over-quoting of module constants.
        # Templates like cv2.cvtColor({src}, '{code}') produce 'cv2.COLOR_BGR2GRAY'
        # when {code} resolves to cv2.COLOR_BGR2GRAY. Remove extraneous quotes.
        transformed = re.sub(
            r"['\"]([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            r"\1",
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
            "pandas": "pandas", "numpy": "numpy",
            "scipy": "scipy", "sklearn": "sklearn",
            "matplotlib": "matplotlib", "json": "json",
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

        used_names: Set[str] = set()
        for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

        required_imports: Set[str] = set()
        for imp in existing_imports:
            mod_root = (
                imp.replace("import", "").strip().split()[0].split(".")[0].lower()
                if imp.startswith("import") else imp.split(".")[0].lower()
            )
            if mod_root in UnificationGate.DENYLISTED_IMPORTS:
                continue
            required_imports.add(imp)

        for name in used_names:
            if name in UnificationGate.CANONICAL_IMPORTS:
                required_imports.add(UnificationGate.CANONICAL_IMPORTS[name])
            elif name in sys.stdlib_module_names:
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
