# main.py
from router import LatticeRouter
from unification import ExecutionContext, UnificationGate


def run_nstl_engine(
    user_prompt: str, seed_var_name: str, seed_type: str, seed_state: str
):
    print("\n" + "=" * 70)
    print(f" NSTL ENGINE RUNNING FOR INTENT: '{user_prompt}'")
    print("=" * 70)

    context = ExecutionContext()
    context.declare_variable(name=seed_var_name, var_type=seed_type, state=seed_state)
    print(
        f"[SEED INITIALIZED] Created {seed_var_name} (Type: {seed_type}, State: {seed_state})"
    )

    router = LatticeRouter()
    execution_path = router.plan_path(user_prompt)

    compiled_pipeline = []
    variable_counter = int(seed_var_name.split("_")[1]) + 1

    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell, variable_counter)

        # If a block returns None, it simply means this path variant
        # is incompatible with our types, so we skip it safely
        if code_block is not None:
            compiled_pipeline.append(code_block)
            variable_counter += 1

    print("\n" + "-" * 40)
    print(" GENERATED COHESIVE SYSTEM ASSEMBLY:")
    print("-" * 40)
    print("\n".join(compiled_pipeline))
    print("=" * 70)


if __name__ == "__main__":
    # Case A: Testing modern Pandas flow manipulation
    prompt_a = (
        "Read my data csv file, clean out any nan rows, and sort the data rows securely"
    )
    run_nstl_engine(
        prompt_a, seed_var_name="var_0", seed_type="str", seed_state="filepath"
    )

    # Case B: Testing core Computer Science Data Structure / Algorithms pipeline
    prompt_b = "Take this messy collection list, sort it order wise, and find a item location within it"
    run_nstl_engine(
        prompt_b,
        seed_var_name="var_0",
        seed_type="list",
        seed_state="unordered_collection",
    )
