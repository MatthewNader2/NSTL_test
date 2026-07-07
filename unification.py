# unification.py
import os
import re
from typing import Optional

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
        quoted_items = re.findall(r'["\'](["\']+)["\']', user_prompt)
        if quoted_items:
            self.extracted_parameters["explicit_filename"] = quoted_items[0]
        else:
            file_match = re.search(
                r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html))\b",
                user_prompt.lower(),
            )
            if file_match:
                self.extracted_parameters["explicit_filename"] = file_match.group(1)

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
        # FIX: Iterate in REVERSE to grab the most recently generated variable in scope!
        # Strict mathematical type validation against AlgebraicSignature
        for var_name, current_signature in reversed(list(self.registry.items())):
            if current_signature.matches(expected_signature):
                return var_name
        return None


class UnificationGate:
    """Performs dynamic monadic structural unification across cell signatures."""

    @staticmethod
    def unify(context: ExecutionContext, target_cell) -> str:
        matching_input_var = context.find_compatible_variable(target_cell.inputs)

        if not matching_input_var:
            if context.registry:
                matching_input_var = list(context.registry.keys())[-1]
            else:
                matching_input_var = "input_source"

        raw_output_name = target_cell.outputs.state.lower().strip() or "output_var"
        output_var_name = context.declare_variable(
            name=raw_output_name,
            signature=target_cell.outputs,
        )

        # Handle either MicroCell or MacroCell (mostly MicroCell for code snippets)
        compiled_snippet = getattr(target_cell, "code_template", "")
        if compiled_snippet:
            compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
            compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

            if "explicit_filename" in context.extracted_parameters:
                user_assigned_name = context.extracted_parameters["explicit_filename"]
                compiled_snippet = re.sub(
                    r"['\"]export\.(?:csv|json|html|feather|parquet)['\"]\",",
                    f"'{user_assigned_name}'",
                    compiled_snippet,
                )
                compiled_snippet = compiled_snippet.replace(
                    "export.csv", user_assigned_name
                )

        print(
            f"[UNIFICATION SUCCESS] Linked {matching_input_var} -> {target_cell.cell_id} -> {output_var_name}"
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
            print(f"[UNIFICATION ERROR] Synthesized typestates do not match. Expected {expected_inputs}->{expected_outputs}, got {in_type}->{out_type}")
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

        # Append and save
        data["cells"].append(synthesized_dict)

        # BUG 16 FIX: Write atomically via a temp file + os.replace() so a mid-write
        # crash cannot corrupt the cache (which previously lost ALL synthesized nodes).
        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"[UNIFICATION CACHE ERROR] Failed to save to {cache_path}: {e}")
            # Clean up orphaned temp file if it exists
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

        print(f"[UNIFICATION CACHE] Successfully saved {synthesized_dict.get('cell_id')} to {cache_path}")
        return True
