# main.py
from lattice import DomainLattice, LatticeOrchestrator, MicroCell
from router import LatticeRouter
from unification import ExecutionContext, UnificationGate

# ==========================================
# DEFINE COMPONENT TREES INDEPENDENTLY
# ==========================================

# Tree A: Standard Python Collections Operations
standard_tree = DomainLattice("Standard_Python_Builtins")
standard_tree.add_cell(
    MicroCell(
        cell_id="SORT_LIST",
        stage=1,
        keywords={"sort", "order", "list", "array"},
        code_template="{output_var} = sorted({input_var})",
        inputs={"input_type": "list", "expected_state": "unordered_collection"},
        outputs={"output_type": "list", "resulting_state": "sorted_collection"},
    )
)

# Tree B: Specialized Pandas Library
pandas_tree = DomainLattice("Pandas_Dataframe_Library")
pandas_tree.add_cell(
    MicroCell(
        cell_id="READ_CSV",
        stage=1,
        keywords={"read", "csv", "load", "file"},
        code_template="with open('{input_var}', 'r') as f:\n    {output_var} = pd.read_csv(f)",
        inputs={"input_type": "str", "expected_state": "filepath"},
        outputs={"output_type": "DataFrame", "resulting_state": "raw_data"},
    )
)
pandas_tree.add_cell(
    MicroCell(
        cell_id="FILTER_NAN",
        stage=2,
        keywords={"clean", "nan", "drop", "null"},
        code_template="{output_var} = {input_var}.dropna()",
        inputs={"input_type": "DataFrame", "expected_state": "raw_data"},
        outputs={"output_type": "DataFrame", "resulting_state": "clean_data"},
    )
)


# ==========================================
# RUNTIME EXPERIMENT PIPELINE
# ==========================================


def execute_system(prompt: str, orchestrator: LatticeOrchestrator):
    context = ExecutionContext()
    # Seed initial environment configuration assuming a file stream entry
    context.declare_variable(name="var_0", var_type="str", state="filepath")

    router = LatticeRouter(orchestrator)
    execution_path = router.plan_path(prompt)

    compiled_pipeline = []
    variable_counter = 1

    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell, variable_counter)
        if code_block is not None:
            compiled_pipeline.append(code_block)
            variable_counter += 1

    print("\n[RESULTING ASSEMBLY OUTPUT]:")
    print(
        "\n".join(compiled_pipeline)
        if compiled_pipeline
        else "[EMPTY] No valid path generated."
    )


if __name__ == "__main__":
    print("Initializing Wrapper Framework Experiment...\n")

    # 1. Initialize the central Wrapper
    wrapper = LatticeOrchestrator()

    # 2. Mount ONLY the standard tree first
    wrapper.register_domain_tree(standard_tree)

    # Test Prompt that requires Pandas functionality
    user_prompt = "read a database csv file and clean out the nan cells"

    print(
        "\n--- TEST 1: Requesting Pandas actions while only Standard Tree is mounted ---"
    )
    execute_system(user_prompt, wrapper)

    print("\n" + "=" * 70)

    # 3. Dynamically plug in the Pandas Tree live
    print("--- DYNAMIC CAPABILITY UPGRADE ---")
    wrapper.register_domain_tree(pandas_tree)

    print("\n--- TEST 2: Running the exact same prompt after mounting Pandas Tree ---")
    execute_system(user_prompt, wrapper)
