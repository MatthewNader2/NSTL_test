# unification.py


class ExecutionContext:
    """Tracks the live state of variables declared during the pipeline execution."""

    def __init__(self):
        # Format: { variable_name: {"type": str, "state": str} }
        self.registry = {}

    def declare_variable(self, name: str, var_type: str, state: str):
        self.registry[name] = {"type": var_type, "state": state}

    def find_compatible_variable(self, expected_type: str, expected_state: str) -> str:
        """Looks through active scope to find a variable matching requirements."""
        for var_name, tracking_info in self.registry.items():
            if (
                tracking_info["type"] == expected_type
                and tracking_info["state"] == expected_state
            ):
                return var_name
        return None


class UnificationGate:
    """The safety pass checking type and state compatibility between cells."""

    @staticmethod
    def unify(context: ExecutionContext, target_cell, next_var_id: int) -> str:
        """
        Validates constraints. If successful, binds variable scopes and
        returns the rendered target code snippet. Returns None if validation fails.
        """
        # 1. Structural Check: Find an existing variable that satisfies target cell inputs
        matching_input_var = context.find_compatible_variable(
            expected_type=target_cell.inputs["input_type"],
            expected_state=target_cell.inputs["expected_state"],
        )

        if not matching_input_var:
            print(
                f"[UNIFICATION FAILED] Cell {target_cell.cell_id} cannot resolve inputs."
            )
            return None

        # 2. Dynamic State Creation: Generate an isolated unique output variable name
        output_var_name = f"var_{next_var_id}"

        # 3. Code Generation via Safe Binding
        # Populates the template placeholders safely using the resolved runtime variables
        compiled_snippet = target_cell.code_template.format(
            input_var=matching_input_var, output_var=output_var_name
        )

        # 4. Context Update: Mutate state tracking for subsequent nodes
        context.declare_variable(
            name=output_var_name,
            var_type=target_cell.outputs["output_type"],
            state=target_cell.outputs["resulting_state"],
        )

        print(
            f"[UNIFICATION SUCCESS] Bound {matching_input_var} -> {target_cell.cell_id} -> {output_var_name}"
        )
        return compiled_snippet
