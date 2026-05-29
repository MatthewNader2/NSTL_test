# main.py
from router import LatticeRouter
from unification import ExecutionContext, UnificationGate


def run_nstl_engine(
    user_prompt: str, seed_var_name: str, seed_type: str, seed_state: str
):
    print("\n" + "=" * 70)
    print(f" NSTL ENGINE RUNNING LIVE")
    print("=" * 70)
    print(f"Prompt Input: '{user_prompt}'")
    print(
        f"[SEED CONTEXT] Variable: {seed_var_name} ({seed_type}, State: {seed_state})"
    )
    print("-" * 70)

    context = ExecutionContext()
    context.declare_variable(name=seed_var_name, var_type=seed_type, state=seed_state)

    router = LatticeRouter()
    execution_path = router.plan_path(user_prompt)

    compiled_pipeline = []
    variable_counter = int(seed_var_name.split("_")[1]) + 1

    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell, variable_counter)
        if code_block is not None:
            compiled_pipeline.append(code_block)
            variable_counter += 1

    print("\n" + "-" * 40)
    print(" FINAL VERIFIED SYSTEM ASSEMBLY SCRIPT:")
    print("-" * 40)
    if compiled_pipeline:
        print("\n".join(compiled_pipeline))
    else:
        print(
            "[EMPTY PIPELINE] No valid code could safely pass unification constraint gates."
        )
    print("=" * 70)


if __name__ == "__main__":
    print("NSTL Tokenizer & Unification Engine CLI Active.")
    print("Type 'exit' to quit the runtime loop.\n")

    while True:
        prompt = input("Enter your custom dynamic requirement: ")
        if prompt.lower() == "exit":
            break

        # To test context adaptation, the engine inspects the prompt words
        # to determine whether to seed a file path string or a collection list.
        if (
            "csv" in prompt.lower()
            or "file" in prompt.lower()
            or "database" in prompt.lower()
        ):
            run_nstl_engine(prompt, "var_0", "str", "filepath")
        else:
            run_nstl_engine(prompt, "var_0", "list", "unordered_collection")
        print("\n")
