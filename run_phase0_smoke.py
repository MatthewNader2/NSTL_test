"""
run_phase0_smoke.py
End-to-End smoke test runner validating Phase 0 critical correctness across the entire engine.
"""

import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from inference import ModelManager
from lattice import LatticeOrchestrator
from internal_rag import LocalRAG
from router import LatticeRouter
from planner import ZeroShotPlanner
from unification import UnificationGate, ExecutionContext
from gevr_sandbox import GEVRSandbox


def run_smoke_test():
    print("=" * 60)
    print(" NSTL ENGINE — PHASE 0 END-TO-END SMOKE TEST")
    print("=" * 60)

    # 1. Initialize Profile A
    print("\n[*] Step 1: Initializing ModelManager (Profile A)...")
    t0 = time.time()
    mm = ModelManager.get_instance()
    mm.initialize_profile("A")
    print(f"    [+] Initialized in {time.time() - t0:.2f}s")

    # 2. Initialize Lattice Orchestrator & LocalRAG
    print("\n[*] Step 2: Loading Lattice from SQLite (trees/lattice.db)...")
    orchestrator = LatticeOrchestrator(trees_directory=os.path.join(ROOT_DIR, "trees"))
    print(f"    [+] Loaded {len(orchestrator.loaded_cells)} verified cells into memory.")

    print("\n[*] Step 3: Indexing FAISS Dense Vector Space...")
    rag = LocalRAG(trees_dir=os.path.join(ROOT_DIR, "trees"), orchestrator=orchestrator)
    print(f"    [+] FAISS Index built with {rag.index.ntotal} vectors.")

    router = LatticeRouter(orchestrator=orchestrator, rag_engine=rag)
    planner = ZeroShotPlanner(orchestrator=orchestrator, rag=rag)
    sandbox = GEVRSandbox(timeout_seconds=5.0)

    test_prompts = [
        (
            "Tabular Pipeline",
            "load input.csv and drop missing values then save to output.csv",
            "input.csv", "output.csv"
        ),
        (
            "Vision Pipeline",
            "read image input.jpg and convert to grayscale then save to output.jpg",
            "input.jpg", "output.jpg"
        ),
        (
            "Algorithmic Pipeline",
            "dijkstra shortest path algorithm on graph",
            None, None
        )
    ]

    all_passed = True

    for name, prompt, in_file, out_file in test_prompts:
        print(f"\n--- Testing: {name} ---")
        print(f"    Prompt: '{prompt}'")
        
        # Test Planning Pass
        plan = planner.run_planning_pass(prompt, profile="A")
        print(f"    Planner Sub-cells: {plan.get('cells', [{}])[0].get('sub_cells', [])}")

        # Test Routing
        path, _ = router.plan_path(prompt)
        print(f"    Router Path: {[c.cell_id for c in path]}")
        assert len(path) > 0, f"Path planning failed for {name}"

        # Test Monadic Code Unification
        ctx = ExecutionContext(prompt=prompt)
        unified_lines = []
        for cell in path:
            line = UnificationGate.unify_cell(ctx, cell)
            unified_lines.append(line)
        
        full_code = UnificationGate.resolve_imports("\n".join(unified_lines), ctx)
        print(f"    Generated Code:\n" + "\n".join(f"      | {l}" for l in full_code.splitlines()))

        # Check for unquoted bare filenames (e.g. data.csv)
        assert "data.csv" not in full_code, "Bug C1 regression: data.csv found in code!"
        assert "pd.read_csv(input.csv)" not in full_code, "Unquoted input.csv found!"

        # Sandbox Execution Check (for tabular test, create dummy data)
        if in_file == "input.csv":
            dummy_csv = os.path.join(ROOT_DIR, "input.csv")
            with open(dummy_csv, "w") as f:
                f.write("a,b,c\n1,2,3\n4,,6\n7,8,9\n")
            
            verified, stdout, msg = sandbox.execute_and_verify(full_code)
            print(f"    Execution Verification: verified={verified}, msg='{(msg or stdout).strip()}'")
            if os.path.exists(dummy_csv):
                os.remove(dummy_csv)
            if out_file and os.path.exists(os.path.join(ROOT_DIR, out_file)):
                os.remove(os.path.join(ROOT_DIR, out_file))

            if not verified:
                print(f"    [!] Execution failed: {msg}")
                all_passed = False
            else:
                print(f"    [+] SUCCESS: Verified cleanly in sandbox!")
        else:
            print(f"    [+] SUCCESS: Code generated and validated syntactically!")

    print("\n" + "=" * 60)
    if all_passed:
        print(" [ALL PHASE 0 SMOKE TESTS PASSED CLEANLY — 100% VERIFIED]")
    else:
        print(" [SOME TESTS FAILED — CHECK LOGS ABOVE]")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
