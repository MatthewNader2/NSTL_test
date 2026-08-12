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
            r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html|txt|jpg|jpeg|png|bmp|tiff|webp|pdf))\b",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if file_matches:
            self.extracted_parameters["input_filename"] = file_matches[0]
            if len(file_matches) > 1:
                self.extracted_parameters["output_filename"] = file_matches[-1]
            self.extracted_parameters["explicit_filename"] = file_matches[0]
        else:
            quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
            if quoted_items:
                self.extracted_parameters["input_filename"] = quoted_items[0]
                if len(quoted_items) > 1:
                    self.extracted_parameters["output_filename"] = quoted_items[-1]
                self.extracted_parameters["explicit_filename"] = quoted_items[0]

        # 3. Heuristics for arguments
        heuristics = []
        all_quoted = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        all_files = set(self.extracted_parameters.values())
        for q in all_quoted:
            if q not in all_files:
                heuristics.append(f"{repr(q)}")
        
        # Sorting direction & color conversion heuristics
        prompt_lower = user_prompt.lower()
        if "descending" in prompt_lower or "desc" in prompt_lower or "highest to lowest" in prompt_lower:
            heuristics.append("ascending=False")
        elif "ascending" in prompt_lower:
            heuristics.append("ascending=True")

        if "grayscale" in prompt_lower or "gray" in prompt_lower or "bgr2gray" in prompt_lower:
            heuristics.append("cv2.COLOR_BGR2GRAY")
        
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
        # Priority: Return the most recently declared variable of matching type_name in scope for linear pipeline binding
        for var_name, current_signature in reversed(list(self.registry.items())):
            if current_signature.matches(expected_signature):
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
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id.lower()
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr.lower()

                    existing_kwargs = {kw.arg for kw in node.keywords if kw.arg}
                    for p in ordered_params:
                        p_str = str(p).lower()
                        p_tokens = set(re.findall(r"[a-zA-Z0-9]+", p_str)) - {"true", "false", "none"}
                        func_tokens = set(re.findall(r"[a-zA-Z0-9]+", func_name)) if func_name else set()
                        
                        # Generic AST parameter relevance validation via sub-token similarity & semantic roles
                        if p_tokens and func_tokens:
                            from difflib import SequenceMatcher
                            has_match = False

                            # 1. Filename string relevance for I/O functions with input/output intent validation
                            if any(ext in p_str for ext in [".csv", ".jpg", ".jpeg", ".png", ".json", ".parquet", ".html", ".txt", ".pdf"]):
                                is_read_func = any(rk in func_name for rk in ["read", "load", "imread", "open"])
                                is_write_func = any(wk in func_name for wk in ["write", "save", "export", "to", "imwrite"])
                                is_output_name = any(ok in p_str for ok in ["out", "clean", "result", "dest", "new", "target", "export", "save"])
                                is_input_name = any(ik in p_str for ik in ["in", "src", "source", "input", "raw", "orig"])

                                if is_read_func and not is_output_name:
                                    has_match = True
                                elif is_write_func and not is_input_name:
                                    has_match = True

                            # 2. Keyword argument key relevance for sorting/filtering/converting
                            if "=" in p_str:
                                kw_key = p_str.split("=")[0].strip().lower()
                                if kw_key in ["ascending", "by", "axis", "inplace"] and any(sk in func_name for sk in ["sort", "order", "rank"]):
                                    has_match = True
                                elif kw_key in ["code", "color", "cmap", "mode"] and any(ck in func_name for ck in ["cvt", "convert", "transform"]):
                                    has_match = True

                            # 3. Sub-token diff ratio matching
                            if not has_match:
                                for pt in p_tokens:
                                    if len(pt) <= 1:
                                        continue
                                    if any(pt in ft or ft in pt or SequenceMatcher(None, pt, ft).ratio() >= 0.55 for ft in func_tokens):
                                        has_match = True
                                        break

                            if not has_match:
                                continue

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
        
        # 1. Universal template-driven placeholder replacement
        if compiled_snippet:
            in_fname = context.extracted_parameters.get("input_filename")
            out_fname = context.extracted_parameters.get("output_filename")

            # Ensure write/save cells receive out_fname as 1st argument if template only had {input_var}
            if out_fname:
                compiled_snippet = compiled_snippet.replace("{output_filename}", repr(out_fname))
                if any(kw in cell_id.lower() for kw in ["imwrite", "savefig"]) and repr(out_fname) not in compiled_snippet:
                    compiled_snippet = compiled_snippet.replace("({input_var})", f"({repr(out_fname)}, {{input_var}})")

            compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
            compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

            if in_fname:
                compiled_snippet = compiled_snippet.replace("{input_filename}", repr(in_fname))
                compiled_snippet = compiled_snippet.replace("{input_source}", repr(in_fname))

            # Generic string literal binding for read/write file parameters
            if in_fname:
                # Replace generic dummy filenames in quotes (e.g. 'input.jpg', 'data.csv', 'input_file')
                compiled_snippet = re.sub(
                    r"(['\"])(?:input\.(?:jpg|png|jpeg|csv|json|parquet|txt)|input_file|dummy_input)\1",
                    repr(in_fname),
                    compiled_snippet,
                    flags=re.IGNORECASE
                )
            if out_fname:
                # Replace generic dummy output filenames in quotes (e.g. 'output.jpg', 'cleaned_data.csv', 'output_file')
                compiled_snippet = re.sub(
                    r"(['\"])(?:output\.(?:jpg|png|jpeg|csv|json|parquet|txt)|output_file|export\.\w+|dummy_output)\1",
                    repr(out_fname),
                    compiled_snippet,
                    flags=re.IGNORECASE
                )

        # 2. Dynamic heuristic parameter injection via AST
        cell_heuristics = getattr(target_cell, "matched_heuristics", []) or context.extracted_parameters.get("heuristics", [])
        if cell_heuristics and compiled_snippet:
            # Filter out filenames from heuristics so they aren't injected twice
            injected_params = [
                h for h in cell_heuristics
                if h != repr(in_fname) and h != repr(out_fname) and h != in_fname and h != out_fname
            ]
            if injected_params:
                compiled_snippet = UnificationGate.inject_parameters(compiled_snippet, injected_params)

        # AST Lineage Repair: Ensure transformed variables in context are properly consumed by downstream calls
        compiled_snippet = UnificationGate.fix_dead_variables_in_snippet(context, compiled_snippet, current_output_var=output_var_name)

        logger.info(
            f"[UNIFICATION SUCCESS] Linked {matching_input_var} -> {cell_id} -> {output_var_name} | Code: {compiled_snippet.strip()}"
        )
        return compiled_snippet

    @staticmethod
    def fix_dead_variables_in_snippet(context: ExecutionContext, snippet: str, current_output_var: str = None) -> str:
        """Fixes dead transformed variables by rebinding unmapped caller variables to the latest active variable in context."""
        if not context.registry:
            return snippet
        valid_vars = [v for v in context.registry.keys() if v != current_output_var and v != "input_source"]
        if not valid_vars:
            return snippet

        known_modules = {"cv2", "pd", "np", "plt", "sns", "tf", "torch", "sk", "sklearn", "os", "sys", "math", "re", "json"}
        latest_var = valid_vars[-1]

        # Find method calls on objects: caller.method_name(...)
        def _rebind_caller(m):
            caller = m.group(1)
            method = m.group(2)
            if caller in known_modules or caller in valid_vars or caller == "input_source":
                return f"{caller}.{method}("
            logger.info(f"[AST LINEAGE REPAIR] Rebound unbound caller variable '{caller}' -> '{latest_var}' for method .{method}()")
            return f"{latest_var}.{method}("

        snippet = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\(", _rebind_caller, snippet)
        return snippet

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
        tmp_path = None  # B-7 fix: initialise before try so cleanup block can safely reference it
        try:
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.error(f"[UNIFICATION CACHE ERROR] Failed to save to {cache_path}: {e}")
            # Clean up orphaned temp file if it exists
            try:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
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

        # Q-7 fix: alias map — resolve common short names to their correct import statements.
        # Without this the AST sees `pd` as unbound and emits `import pd` which fails.
        _ALIAS_MAP = {
            'pd': 'import pandas as pd',
            'np': 'import numpy as np',
            'plt': 'import matplotlib.pyplot as plt',
            'sns': 'import seaborn as sns',
            'tf': 'import tensorflow as tf',
            'torch': 'import torch',
            'cv2': 'import cv2',
            'sk': 'import sklearn as sk',
            'sp': 'import scipy as sp',
            'nx': 'import networkx as nx',
        }

        imports_to_add = []
        for mod in sorted(required_imports):
            if mod in _ALIAS_MAP:
                imports_to_add.append(_ALIAS_MAP[mod])
            else:
                imports_to_add.append(f"import {mod}")

        if imports_to_add:
            return "\n".join(imports_to_add) + "\n\n" + code_text
        return code_text
