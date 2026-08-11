# unification.py
import os
import re
from typing import Optional
from log_config import get_logger

logger = get_logger('unification')

# We need AlgebraicSignature imported. It's safe to import it dynamically or statically if lattice.py is in same dir.
from lattice import AlgebraicSignature


class ExecutionContext:
    """Manages variables and literal extraction arguments at runtime using type-state keys."""

    def __init__(self):
        # Format: { variable_name: AlgebraicSignature }
        self.registry: dict[str, AlgebraicSignature] = {}
        self.extracted_parameters = {}

    def extract_prompt_parameters(self, user_prompt: str):
        self.extracted_parameters = {}
        # 1. Extract all filenames with extensions
        file_matches = re.findall(
            r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html|txt))\b",
            user_prompt.lower(),
        )
        if file_matches:
            self.extracted_parameters["input_filename"] = file_matches[0]
            self.extracted_parameters["output_filename"] = file_matches[-1]
            self.extracted_parameters["explicit_filename"] = file_matches[0]
        else:
            quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
            if quoted_items:
                self.extracted_parameters["input_filename"] = quoted_items[0]
                self.extracted_parameters["output_filename"] = quoted_items[-1]
                self.extracted_parameters["explicit_filename"] = quoted_items[0]

        # 3. Heuristics for arguments
        heuristics = []
        all_quoted = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        all_files = set(self.extracted_parameters.values())
        for q in all_quoted:
            if q not in all_files:
                heuristics.append(f"{repr(q)}")
        
        # Sorting direction
        prompt_lower = user_prompt.lower()
        if "descending" in prompt_lower or "desc" in prompt_lower or "highest to lowest" in prompt_lower:
            heuristics.append("ascending=False")
        elif "ascending" in prompt_lower:
            heuristics.append("ascending=True")

        # Find numeric constants
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_prompt)
        for n in numbers:
            heuristics.append(n)
        
        self.extracted_parameters["heuristics"] = heuristics

    def declare_variable(self, name: str, signature: AlgebraicSignature) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        sanitized_name = base_name

        counter = 2
        while sanitized_name in self.registry:
            sanitized_name = f"{base_name}_v{counter}"
            counter += 1

        self.registry[sanitized_name] = signature
        return sanitized_name

    def find_compatible_variable(self, expected_signature: AlgebraicSignature) -> Optional[str]:
        # Pass 1: Look for exact structural match (type_name and state) in reverse
        for var_name, current_signature in reversed(list(self.registry.items())):
            if current_signature.matches(expected_signature):
                return var_name
        # Pass 2: Look for base type_name match in reverse (pipeline continuity)
        for var_name, current_signature in reversed(list(self.registry.items())):
            if current_signature.type_name == "any" or expected_signature.type_name == "any" or current_signature.type_name == expected_signature.type_name:
                return var_name
        return None


class UnificationGate:
    """Performs dynamic monadic structural unification across cell signatures."""

    @staticmethod
    def inject_parameters(code_template: str, parameters: list[str]) -> str:
        """
        Universally injects parameters into a function call code template using AST.
        e.g. output_var = cv2.cvtColor(input_var) -> output_var = cv2.cvtColor(input_var, cv2.COLOR_BGR2GRAY)
        """
        import ast
        try:
            tree = ast.parse(code_template)
            # Ensure positional arguments precede keyword arguments
            pos_params = [p for p in parameters if "=" not in p]
            kw_params = [p for p in parameters if "=" in p]
            ordered_params = pos_params + kw_params

            # Traverse to find the first function call (ast.Call)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    existing_kwargs = {kw.arg for kw in node.keywords if kw.arg}
                    for p in ordered_params:
                        try:
                            dummy_code = f"dummy({p})"
                            dummy_tree = ast.parse(dummy_code)
                            dummy_call = dummy_tree.body[0].value
                            
                            # Transfer positional arguments
                            for arg in dummy_call.args:
                                node.args.append(arg)
                            
                            # Transfer keyword arguments
                            for kwarg in dummy_call.keywords:
                                if kwarg.arg not in existing_kwargs:
                                    node.keywords.append(kwarg)
                                    existing_kwargs.add(kwarg.arg)
                        except Exception as inner_e:
                            logger.warning(f"[AST INJECTION WARNING] Could not parse parameter '{p}': {inner_e}")
                    break
            return ast.unparse(tree)
        except Exception as e:
            logger.error(f"[AST INJECTION ERROR] Failed to manipulate code template: {e}")
            return code_template

    @staticmethod
    def unify(context: ExecutionContext, target_cell, injected_parameters: list[str] = None) -> str:
        matching_input_var = context.find_compatible_variable(target_cell.inputs)

        if not matching_input_var:
            if context.registry:
                matching_input_var = list(context.registry.keys())[-1]
            else:
                matching_input_var = "input_source"

        cell_id = getattr(target_cell, "cell_id", "") or ""
        if cell_id and cell_id != "SYNTHESIZED_NODE":
            parts = cell_id.lower().split('_')
            if len(parts) > 1 and parts[0] in ["pandas", "opencv", "scikit"]:
                parts = parts[1:]
            raw_output_name = "_".join(parts)
        else:
            raw_output_name = getattr(target_cell.outputs, "state", "").lower().strip() or "output_var"
            if raw_output_name == "computed":
                raw_output_name = "computed_var"
        
        output_var_name = context.declare_variable(
            name=raw_output_name,
            signature=target_cell.outputs,
        )

        compiled_snippet = getattr(target_cell, "code_template", "")
        
        # 1. Literal templating (must happen before AST parsing so the syntax is valid)
        if compiled_snippet:
            compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
            compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

            out_fname = context.extracted_parameters.get("output_filename") or context.extracted_parameters.get("explicit_filename")
            if out_fname and ("export.csv" in compiled_snippet or "to_csv" in compiled_snippet):
                compiled_snippet = re.sub(
                    r"(['\"])export\.(csv|json|html|feather|parquet)\1",
                    repr(out_fname),
                    compiled_snippet,
                )
                compiled_snippet = compiled_snippet.replace(
                    "export.csv", out_fname
                )
                if ".to_csv()" in compiled_snippet:
                    compiled_snippet = compiled_snippet.replace(
                        ".to_csv()", f".to_csv({out_fname!r}, index=False)"
                    )
                
        # 2. Targeted parameter filtering by cell ID
        cell_id_lower = cell_id.lower()
        cell_specific_params = []

        if "sort" in cell_id_lower:
            for h in context.extracted_parameters.get("heuristics", []):
                if "ascending" in h or h.startswith("'") or h.startswith('"'):
                    if h not in cell_specific_params:
                        cell_specific_params.append(h)
        elif "to_csv" in cell_id_lower or "export" in cell_id_lower or "read_csv" in cell_id_lower or "read" in cell_id_lower:
            # Read/Write nodes handle parameters via literal templating above
            cell_specific_params = []
        else:
            cell_heuristics = getattr(target_cell, "matched_heuristics", [])
            for h in cell_heuristics:
                if h not in cell_specific_params:
                    cell_specific_params.append(h)

        if cell_specific_params and compiled_snippet:
            compiled_snippet = UnificationGate.inject_parameters(compiled_snippet, cell_specific_params)

        logger.info(
            f"[UNIFICATION SUCCESS] Linked {matching_input_var} -> {cell_id} -> {output_var_name}"
        )
        return compiled_snippet

    @staticmethod
    def validate_synthesis(
        synthesized_dict: dict,
        expected_inputs: str,
        expected_outputs: str,
        trees_dir: str = "trees",
    ) -> bool:
        """
        Validates the synthesized MicroCell JSON against the required typestates.
        If valid, caches it permanently.
        """
        import json

        # 1. Typestate Match
        inputs = synthesized_dict.get("inputs", {})
        outputs = synthesized_dict.get("outputs", {})

        if not isinstance(inputs, dict):
            inputs = {}
        if not isinstance(outputs, dict):
            outputs = {}

        # Support both old-schema (input_type/output_type) and new-schema (type_name) keys
        in_type  = inputs.get("type_name",  inputs.get("input_type",  ""))
        out_type = outputs.get("type_name", outputs.get("output_type", ""))

        if in_type != expected_inputs or out_type != expected_outputs:
            logger.error(f"[UNIFICATION ERROR] Synthesized typestates do not match. Expected {expected_inputs}->{expected_outputs}, got {in_type}->{out_type}")
            return False

        # 2. Permanent Cache to <trees_dir>/micro/synthesized_nodes.json
        # BUG 15 FIX: Use the provided trees_dir instead of the hardcoded relative path
        #             so frozen/PyInstaller builds and different CWDs work correctly.
        cache_dir  = os.path.join(trees_dir, "micro")
        cache_path = os.path.join(cache_dir, "synthesized_nodes.json")
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"domain_name": "Synthesized_Domain", "cells": []}
        else:
            data = {"domain_name": "Synthesized_Domain", "cells": []}

        # Append and save, replacing any previous synthesized cell with the same ID
        # so repeated retries do not bloat the cache or create duplicate topology entries.
        new_cell_id = synthesized_dict.get("cell_id")
        if new_cell_id:
            data["cells"] = [
                cell for cell in data.get("cells", [])
                if not isinstance(cell, dict) or cell.get("cell_id") != new_cell_id
            ]
        data.setdefault("cells", []).append(synthesized_dict)

        # BUG 16 FIX: Write atomically via a temp file + os.replace() so a mid-write
        # crash cannot corrupt the cache (which previously lost ALL synthesized nodes).
        # Use tempfile.mkstemp for unique names to prevent concurrent thread collisions.
        import tempfile
        try:
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.error(f"[UNIFICATION CACHE ERROR] Failed to save to {cache_path}: {e}")
            # Clean up orphaned temp file if it exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            return False

        logger.info(f"[UNIFICATION CACHE] Successfully saved {synthesized_dict.get('cell_id')} to {cache_path}")
        return True

    @staticmethod
    def resolve_imports(code_text: str, context: 'ExecutionContext' = None) -> str:
        """Dynamically detects unbound top-level module variables and prepends their imports."""
        import ast
        import builtins

        try:
            tree = ast.parse(code_text)
        except SyntaxError as e:
            logger.error(f"[AST IMPORT ERROR] Could not parse code for import resolution: {e}")
            return code_text

        loaded_names = set()
        stored_names = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    loaded_names.add(node.id)
                elif isinstance(node.ctx, ast.Store):
                    stored_names.add(node.id)
                self.generic_visit(node)
            
            def visit_arg(self, node):
                stored_names.add(node.arg)
                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    stored_names.add(alias.asname or alias.name)
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    stored_names.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                stored_names.add(node.name)
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                stored_names.add(node.name)
                self.generic_visit(node)

        visitor = NameVisitor()
        visitor.visit(tree)

        builtin_names = set(dir(builtins))
        context_vars = set(context.registry.keys()) if context else set()
        context_vars.add("input_source") # NEVER try to import this
        
        # We also want to ignore pandas, np, etc if they are already imported elsewhere, but the AST visitor handles `stored_names`
        # which includes imports. However, if the code uses `pd` but `pd` isn't imported, it might prepend `import pd` which will fail.
        # But this is a generic resolver. Let's fix the `input_source` issue first.
        required_imports = loaded_names - stored_names - builtin_names - context_vars

        imports_to_add = []
        for mod in sorted(list(required_imports)):
            imports_to_add.append(f"import {mod}")

        if imports_to_add:
            return "\n".join(imports_to_add) + "\n\n" + code_text
        return code_text
