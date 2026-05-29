# main.py
from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import ExecutionContext, UnificationGate


def execute_production_engine(user_prompt: str, initial_file: str):
    print("\n" + "=" * 70)
    print(" NSTL ENGINE CORE COMPILED AND RUNNING")
    print("=" * 70)

    # 1. Initialize Orchestrator Wrapper - Spawns automatic file discovery pass
    wrapper = LatticeOrchestrator()

    # 2. Seed Environment Scope Context
    context = ExecutionContext()
    context.declare_variable(name="var_0", var_type="str", state="filepath")
    print(f"\n[SEED PROFILE] Bound variable var_0 = '{initial_file}' (type: str)")

    # 3. Prompt Analysis via Tokenizer Route Matrix
    router = LatticeRouter(wrapper)
    execution_path = router.plan_path(user_prompt)

    # 4. Sequential Type-Verification Gate Compilation Passing
    compiled_pipeline = []
    variable_counter = 1

    print("\n--- Processing Monadic Unification Structural Safety Verification ---")
    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell, variable_counter)
        if code_block is not None:
            compiled_pipeline.append(code_block)
            variable_counter += 1

    print("\n" + "=" * 70)
    print(" PRODUCTION APPLICATION OUTPUT PIPELINE GENERATED SUCCESSFULLY")
    print("=" * 70)
    if compiled_pipeline:
        print("\n".join(compiled_pipeline))
    else:
        print(
            "[SYSTEM ALERT] Incompatible path sequence or missing active domain module capability."
        )
    print("=" * 70)


if __name__ == "__main__":
    # Realistic complex client prompt requesting an end-to-end data processing stream
    realistic_prompt = "load my source data file, clean out any missing values, format the column headers to lowercase, and give me a full summary of the statistical metrics to export to json format"

    execute_production_engine(realistic_prompt, initial_file="raw_sensor_telemetry.csv")
