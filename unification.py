# unification.py


class ExecutionContext:
    def __init__(self):
        self.registry = {}

    def declare_variable(self, name: str, var_type: str, state: str):
        self.registry[name] = {"type": var_type, "state": state}

    def find_compatible_variable(self, expected_type: str, expected_state: str) -> str:
        for var_name, tracking_info in self.registry.items():
            if (
                tracking_info["type"] == expected_type
                and tracking_info["state"] == expected_state
            ):
                return var_name
        return None


class UnificationGate:
    @staticmethod
    def unify(context: ExecutionContext, target_cell, next_var_id: int) -> str:
        # Check if an active variable fits the cell constraints
        matching_input_var = context.find_compatible_variable(
            expected_type=target_cell.inputs["input_type"],
            expected_state=target_cell.inputs["expected_state"],
        )

        # Polymorphic protection layer: skip if mismatch found
        if not matching_input_var:
            return None

        output_var_name = f"var_{next_var_id}"
        compiled_snippet = target_cell.code_template.format(
            input_var=matching_input_var, output_var=output_var_name
        )

        context.declare_variable(
            name=output_var_name,
            var_type=target_cell.outputs["output_type"],
            state=target_cell.outputs["resulting_state"],
        )

        print(
            f"[UNIFICATION SUCCESS] Bound {matching_input_var} -> {target_cell.cell_id} -> {output_var_name}"
        )
        return compiled_snippet
