# unification.py
import re


class ExecutionContext:
    """Manages variables and literal extraction arguments at runtime using type-state keys."""

    def __init__(self):
        # Format: { variable_name: {"type": str, "state": str} }
        self.registry = {}
        self.extracted_parameters = {}

    def extract_prompt_parameters(self, user_prompt: str):
        self.extracted_parameters = {}
        quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        if quoted_items:
            self.extracted_parameters["explicit_filename"] = quoted_items[0]
        else:
            file_match = re.search(
                r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html))\b",
                user_prompt.lower(),
            )
            if file_match:
                self.extracted_parameters["explicit_filename"] = file_match.group(1)

    def declare_variable(self, name: str, var_type: str, state: str) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        sanitized_name = base_name

        counter = 2
        while sanitized_name in self.registry:
            sanitized_name = f"{base_name}_v{counter}"
            counter += 1

        self.registry[sanitized_name] = {"type": var_type, "state": state}
        return sanitized_name

    def find_compatible_variable(self, expected_type: str, expected_state: str) -> str:
        expected_type_clean = expected_type.lower().strip()
        expected_state_clean = (
            expected_state.lower().replace("_", "").replace("-", "").strip()
        )

        # FIX: Iterate in REVERSE to grab the most recently generated variable in scope!
        for var_name, tracking_info in reversed(list(self.registry.items())):
            current_type = tracking_info["type"].lower().strip()
            current_state = (
                tracking_info["state"].lower().replace("_", "").replace("-", "").strip()
            )

            if current_type == expected_type_clean and (
                expected_state_clean in current_state
                or current_state in expected_state_clean
            ):
                return var_name
        return None


class UnificationGate:
    """Performs dynamic monadic structural unification across cell signatures."""

    @staticmethod
    def unify(context: ExecutionContext, target_cell) -> str:
        matching_input_var = context.find_compatible_variable(
            expected_type=target_cell.inputs["input_type"],
            expected_state=target_cell.inputs["expected_state"],
        )

        if not matching_input_var:
            if context.registry:
                matching_input_var = list(context.registry.keys())[-1]
            else:
                matching_input_var = "input_source"

        raw_output_name = target_cell.outputs["resulting_state"].lower().strip()
        output_var_name = context.declare_variable(
            name=raw_output_name,
            var_type=target_cell.outputs["output_type"],
            state=target_cell.outputs["resulting_state"],
        )

        compiled_snippet = target_cell.code_template
        compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
        compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

        if "explicit_filename" in context.extracted_parameters:
            user_assigned_name = context.extracted_parameters["explicit_filename"]
            compiled_snippet = re.sub(
                r"['\"]export\.(?:csv|json|html|feather|parquet)['\"]",
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
