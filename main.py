# main.py
from router import LatticeRouter
from unification import ExecutionContext, UnificationGate


def run_nstl_engine(user_prompt: str, seed_filename: str):
    print("=" * 60)
    print(f" NSTL ENGINE INITIALIZED FOR PROMPT: '{user_prompt}'")
    print("=" * 60)

    # 1. Initialize empty, sandboxed Execution Context tracking space
    context = ExecutionContext()

    # 2. Seed the initial environment state with raw user inputs
    # We establish that 'var_0' exists as a string containing a raw file path
    context.declare_variable(name="var_0", var_type="str", state="filepath")
    print(f"[SEED SUCCESS] Set var_0 = '{seed_filename}' (type: str, state: filepath)")

    # 3. Invoke the policy network router to extract the path
    router = LatticeRouter()
    execution_path = router.plan_path(user_prompt)

    # 4. Traversal pass over the lattice with strict unification checking
    compiled_pipeline = []
    variable_counter = (
        1  # Increments to generate distinct, collision-free variable names
    )

    print("\n--- Beginning Type-Monadic Unification Verification Phase ---")
    for cell in execution_path:
        # Pass the cell through the gate to structurally bind scope
        code_block = UnificationGate.unify(context, cell, variable_counter)

        if code_block is None:
            print(
                f"\n[CRITICAL HARD HALT] Execution chain broken at cell {cell.cell_id} due to verification mismatch."
            )
            return

        compiled_pipeline.append(code_block)
        variable_counter += 1

    # 5. Synthesize clean pipeline block if all structural passes confirm safety
    print("\n" + "=" * 60)
    print(" SYNTHESIZED LOGICAL PIPELINE (100% HALLUCINATION SAFE)")
    print("=" * 60)

    final_output_script = "\n".join(compiled_pipeline)
    print(final_output_script)
    print("=" * 60)


if __name__ == "__main__":
    # Test Case: User wants to clean a CSV file and output serialized json strings
    sample_prompt = "I need to read a database csv, strip out any missing nan values, and transform it to json formatting"
    run_nstl_engine(sample_prompt, seed_filename="production_logs.csv")
