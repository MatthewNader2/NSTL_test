"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate, Parameter Binding, and AST Code Synthesizer.
"""

from __future__ import annotations
import ast
import os
import re
import sys
import tempfile
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

    def __init__(self, prompt: str = ""):
        self._scope: Dict[str, VariableBinding] = {}
        self.declared_dependencies: Set[str] = set()
        self._var_counter: Dict[str, int] = {}
        self.prompt_hint: str = prompt
        if prompt:
            self._extract_prompt_literals(prompt)

    def _extract_prompt_literals(self, prompt: str):
        """Extracts filenames, column names, and sorting flags from user prompt."""
        # 1. Extract file paths with known extensions
        files = re.findall(r'[\'"]?([a-zA-Z0-9_\-./]+\.(?:csv|json|parquet|xlsx|jpg|jpeg|png|bmp|txt|db|h5))[\'"]?', prompt, re.IGNORECASE)
        files = list(dict.fromkeys(files))

        if files:
            self.declare_variable("input_file", AlgebraicSignature("str", "source_identifier"), literal_value=repr(files[0]))
            if len(files) > 1:
                self.declare_variable("output_file", AlgebraicSignature("str", "dest_identifier"), literal_value=repr(files[-1]))

        # 2. Extract column/attribute names (e.g. "by the 'age' column")
        col_match = re.search(r'(?:column|col|by|sort\s+by)\s+(?:the\s+|a\s+|an\s+)*[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?', prompt, re.IGNORECASE)
        if col_match:
            col_name = col_match.group(1)
            if col_name.lower() not in ("descending", "ascending", "the", "a", "an", "column", "columns", "data", "file", "csv"):
                self.declare_variable("by_col", AlgebraicSignature("str", "column_name"), literal_value=repr(col_name))

        # 3. Extract sorting direction
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
        self._scope[var_name] = VariableBinding(name=var_name, signature=signature, literal_value=literal_value, lineage_parent=parent_var)
        return var_name

    def find_compatible_variable(self, required_signature: AlgebraicSignature) -> Optional[VariableBinding]:
        """Finds the most recently declared variable in scope whose typestate unifies."""
        for var_name, binding in reversed(list(self._scope.items())):
            if binding.signature.unifies_with(required_signature):
                return binding
        return None

    def get_latest_data_variable(self) -> Optional[str]:
        """Returns the latest in-scope non-metadata variable."""
        for var_name, binding in reversed(list(self._scope.items())):
            if binding.signature.type_name not in ("str", "bool", "int", "None"):
                return var_name
        return list(self._scope.keys())[-1] if self._scope else None


class UnificationGate:
    """Monadic code unifier: binds cell ports and templates to in-scope variables and prompt literals."""

    @staticmethod
    def unify_cell(
        context: ExecutionContext,
        cell: Cell,
        explicit_arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        explicit_args = explicit_arguments or {}

        # 1. Resolve Primary Data Input Variable
        primary_input_var = context.get_latest_data_variable()

        # 2. Allocate Output Variable in Context
        cell_id_clean = cell.cell_id.lower().replace("_cell", "").replace("_default", "")
        raw_out_name = cell_id_clean.split('_')[-1]
        output_var_name = context.declare_variable(
            base_name=f"{raw_out_name}_out",
            signature=cell.primary_output,
            parent_var=primary_input_var
        )

        # 3. Track Real Dependencies
        if cell.dependencies:
            for dep in cell.dependencies:
                dep_str = str(dep).strip()
                if dep_str and not any(dep_str.startswith(f"import {fake}") for fake in [
                    "generic", "algorithms", "data_engineering", "machine_learning",
                    "python_core", "data_processing", "image_processing", "nlp"
                ]):
                    context.declared_dependencies.add(dep_str)

        raw_code = cell.code_template
        if not raw_code:
            parts = cell.cell_id.lower().replace("_cell", "").replace("_default", "").split("_")
            domain_l = (cell.domain_name or "").lower()

            if domain_l in ("opencv", "cv2"):
                mod = "cv2"
            elif domain_l in ("pandas", "numpy", "scipy", "sklearn", "matplotlib", "json", "math", "os", "sys"):
                mod = domain_l
            else:
                mod = ""  # No module prefix for generic/python_core/algorithms

            func = "_".join(parts[1:]) if len(parts) > 1 else parts[0]
            arg = primary_input_var or "None"
            if mod:
                return f"{output_var_name} = {mod}.{func}({arg})"
            return f"{output_var_name} = {func}({arg})"

        # 4. Universal AST Placeholder Resolution
        transformed = raw_code.replace("{output_var}", output_var_name)

        # Map all variants of primary data input placeholders
        for in_ph in ("{input_var}", "{df}", "{src}", "{img}", "{image}", "{data}", "{graph}", "{array}", "{mat}"):
            if in_ph in transformed:
                rep_val = primary_input_var or "None"
                transformed = transformed.replace(in_ph, rep_val)

        # Map source file URI placeholders
        for src_ph in ("{filepath}", "{filename}", "{input_filename}", "{source}", "{input_file}"):
            if src_ph in transformed:
                b = context.find_compatible_variable(AlgebraicSignature("str", "source_identifier"))
                val = b.literal_value if (b and b.literal_value) else "'data.csv'"
                transformed = transformed.replace(src_ph, val)

        # Map destination file URI placeholders
        for dst_ph in ("{dest_path}", "{output_filename}", "{destination}", "{output_file}", "{dest}"):
            if dst_ph in transformed:
                b = context.find_compatible_variable(AlgebraicSignature("str", "dest_identifier"))
                val = b.literal_value if (b and b.literal_value) else "'output.csv'"
                transformed = transformed.replace(dst_ph, val)

        # Map column identifier placeholders
        if "{by}" in transformed or "{by_column}" in transformed:
            b = context.find_compatible_variable(AlgebraicSignature("str", "column_name"))
            val = b.literal_value if (b and b.literal_value) else (f"{primary_input_var}.columns[0]" if primary_input_var else "0")
            transformed = transformed.replace("{by}", val).replace("{by_column}", val)

        # Map sort direction flags
        if "{ascending}" in transformed:
            b = context.find_compatible_variable(AlgebraicSignature("bool", "sort_flag"))
            val = b.literal_value if (b and b.literal_value) else "True"
            transformed = transformed.replace("{ascending}", val)

        # Map start/goal node identifiers
        for node_ph in ("{start}", "{start_node}"):
            if node_ph in transformed:
                transformed = transformed.replace(node_ph, "'A'")

        # Map math / binary arguments
        if "{a}" in transformed and "{b}" in transformed:
            transformed = transformed.replace("{a}", "5").replace("{b}", "7")

        # Fallback for remaining arbitrary port placeholders
        for ph in re.findall(r"\{([a-zA-Z0-9_]+)\}", transformed):
            if ph in explicit_args:
                transformed = transformed.replace(f"{{{ph}}}", repr(explicit_args[ph]))
            else:
                transformed = transformed.replace(f"{{{ph}}}", "None")

        logger.info(f"[UNIFICATION SUCCESS] Bound {cell.cell_id} -> {output_var_name}")
        return transformed

    @staticmethod
    def resolve_imports(code_text: str, context: Optional[ExecutionContext] = None) -> str:
        """Collects top-level imports and deduplicates them into a clean PEP 8 header."""
        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return code_text

        top_imports = set()
        if context:
            top_imports.update(context.declared_dependencies)

        # Inspect AST names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

        canonical = {
            "pd": "import pandas as pd",
            "pandas": "import pandas as pd",
            "np": "import numpy as np",
            "numpy": "import numpy as np",
            "cv2": "import cv2",
            "plt": "import matplotlib.pyplot as plt",
            "heapq": "import heapq",
            "json": "import json",
            "os": "import os",
            "sys": "import sys",
            "math": "import math"
        }

        for name in used_names:
            if name in canonical:
                top_imports.add(canonical[name])
            elif name in sys.stdlib_module_names:
                top_imports.add(f"import {name}")

        # Strip invalid non-package imports
        fake_modules = {
            "import generic", "import algorithms", "import data_engineering",
            "import machine_learning", "import python_core", "import data_processing",
            "import image_processing", "import opencv", "import nlp", "import flask"
        }
        top_imports = {imp for imp in top_imports if imp not in fake_modules}

        class ImportStripper(ast.NodeTransformer):
            def visit_Import(self, node):
                return None
            def visit_ImportFrom(self, node):
                return None

        clean_tree = ImportStripper().visit(tree)
        ast.fix_missing_locations(clean_tree)
        clean_body = ast.unparse(clean_tree).strip()

        if top_imports:
            header = "\n".join(sorted(top_imports))
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
            test_code = code.replace("{output_var}", "out").replace("{input_var}", "inp")
            for ph in re.findall(r"\{([a-zA-Z0-9_]+)\}", test_code):
                test_code = test_code.replace(f"{{{ph}}}", "None")
            ast.parse(test_code)
            return True
        except Exception:
            return False
