"""
run_phase0_smoke.py
Hardened End-to-End smoke test runner validating Phase 0 critical correctness across the entire engine.
Enforces strict GEVR sandbox execution and negative assertions on all pipelines.
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from inference import ModelManager
from lattice import LatticeOrchestrator, PortSignature, AlgebraicSignature
from internal_rag import LocalRAG
from router import LatticeRouter
from planner import ZeroShotPlanner
from unification import UnificationGate, ExecutionContext
from gevr_sandbox import GEVRSandbox


def run_smoke_test():
    print("=" * 70)
    print(" NSTL ENGINE — PHASE 0 HARDENED END-TO-END SMOKE TEST")
    print("=" * 70)

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
    print(f"    [+] Input buckets: {len(orchestrator._cells_by_input)}, Output buckets: {len(orchestrator._cells_by_output)}")

    print("\n[*] Step 3: Indexing FAISS Dense Vector Space...")
    rag = LocalRAG(trees_dir=os.path.join(ROOT_DIR, "trees"), orchestrator=orchestrator)
    print(f"    [+] FAISS Index built with {rag.index.ntotal} vectors.")

    router = LatticeRouter(orchestrator=orchestrator, rag_engine=rag)
    planner = ZeroShotPlanner(orchestrator=orchestrator, rag=rag)
    sandbox = GEVRSandbox(timeout_seconds=5.0)

    all_passed = True

    # -------------------------------------------------------------
    # Test 1: Tabular Pipeline (Must execute dropna + read + write)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print(" [1] Tabular Pipeline: Read -> DropNA -> Write")
    print("-" * 50)
    prompt_tab = "load input.csv and drop missing values then save to output.csv"
    print(f"    Prompt: '{prompt_tab}'")

    path_tab, _ = router.plan_path(prompt_tab)
    print(f"    Router Path: {[c.cell_id for c in path_tab]}")

    ctx_tab = ExecutionContext(prompt=prompt_tab)
    lines_tab = [UnificationGate.unify_cell(ctx_tab, c) for c in path_tab]
    code_tab = UnificationGate.resolve_imports("\n".join(lines_tab), ctx_tab)
    print("    Generated Code:\n" + "\n".join(f"      | {l}" for l in code_tab.splitlines()))

    # Assertions
    assert "dropna" in code_tab, "FAIL: dropna was skipped in tabular pipeline!"
    assert "read_csv" in code_tab, "FAIL: read_csv missing in tabular pipeline!"
    assert "to_csv" in code_tab, "FAIL: to_csv missing in tabular pipeline!"
    assert "data.csv" not in code_tab, "FAIL: unquoted data.csv found!"

    # Create test input.csv fixture with missing value
    dummy_csv = os.path.join(ROOT_DIR, "input.csv")
    out_csv = os.path.join(ROOT_DIR, "output.csv")
    with open(dummy_csv, "w") as f:
        f.write("a,b,age\n1,2,25\n4,,30\n7,8,45\n")

    verified_tab, stdout_tab, msg_tab = sandbox.execute_and_verify(code_tab)
    print(f"    Execution: verified={verified_tab}, msg='{(msg_tab or stdout_tab).strip()}'")

    if os.path.exists(out_csv):
        df_out = pd.read_csv(out_csv)
        assert len(df_out) == 2, f"FAIL: dropna did not remove row with NaN! Row count: {len(df_out)}"
        os.remove(out_csv)
        print("    [+] Data Verification: output.csv has exactly 2 rows (missing value dropped)!")
    else:
        print("    [!] ERROR: output.csv was not created!")
        verified_tab = False

    if os.path.exists(dummy_csv):
        os.remove(dummy_csv)

    if not verified_tab:
        print("    [!] Tabular Pipeline FAILED execution.")
        all_passed = False
    else:
        print("    [+] Tabular Pipeline PASSED and VERIFIED cleanly!")

    # -------------------------------------------------------------
    # Test 2: Vision Pipeline (Must execute imread -> cvtColor -> imwrite)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print(" [2] Vision Pipeline: Read -> Grayscale -> Write")
    print("-" * 50)
    prompt_vis = "read image input.jpg and convert to grayscale then save to output.jpg"
    print(f"    Prompt: '{prompt_vis}'")

    path_vis, _ = router.plan_path(prompt_vis)
    print(f"    Router Path: {[c.cell_id for c in path_vis]}")

    ctx_vis = ExecutionContext(prompt=prompt_vis)
    lines_vis = [UnificationGate.unify_cell(ctx_vis, c) for c in path_vis]
    code_vis = UnificationGate.resolve_imports("\n".join(lines_vis), ctx_vis)
    print("    Generated Code:\n" + "\n".join(f"      | {l}" for l in code_vis.splitlines()))

    # Assertions
    assert "cv2.imread" in code_vis, "FAIL: cv2.imread missing in vision pipeline!"
    assert "cv2.cvtColor" in code_vis, "FAIL: cv2.cvtColor missing in vision pipeline!"
    assert "cv2.imwrite" in code_vis, "FAIL: cv2.imwrite missing in vision pipeline!"
    assert "colorChange" not in code_vis, "FAIL: Injected unwanted cv2.colorChange!"
    assert "print(" not in code_vis, "FAIL: Injected unwanted print stdout!"

    # Create test input.jpg fixture
    dummy_jpg = os.path.join(ROOT_DIR, "input.jpg")
    out_jpg = os.path.join(ROOT_DIR, "output.jpg")
    test_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(dummy_jpg, test_img)

    verified_vis, stdout_vis, msg_vis = sandbox.execute_and_verify(code_vis)
    print(f"    Execution: verified={verified_vis}, msg='{(msg_vis or stdout_vis).strip()}'")

    if os.path.exists(out_jpg):
        saved_img = cv2.imread(out_jpg, cv2.IMREAD_UNCHANGED)
        assert saved_img is not None, "FAIL: output.jpg is corrupt!"
        assert len(saved_img.shape) == 2, f"FAIL: output.jpg is not 2D grayscale! Shape: {saved_img.shape}"
        os.remove(out_jpg)
        print("    [+] Image Verification: output.jpg is a valid 1-channel Grayscale image!")
    else:
        print("    [!] ERROR: output.jpg was not created!")
        verified_vis = False

    if os.path.exists(dummy_jpg):
        os.remove(dummy_jpg)

    if not verified_vis:
        print("    [!] Vision Pipeline FAILED execution.")
        all_passed = False
    else:
        print("    [+] Vision Pipeline PASSED and VERIFIED cleanly!")

    # -------------------------------------------------------------
    # Test 3: Algorithmic Pipeline (Dijkstra without hardcoded dict)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print(" [3] Algorithmic Pipeline: Dijkstra Shortest Path")
    print("-" * 50)
    prompt_algo = "dijkstra shortest path algorithm on graph"
    print(f"    Prompt: '{prompt_algo}'")

    path_algo, _ = router.plan_path(prompt_algo)
    print(f"    Router Path: {[c.cell_id for c in path_algo]}")

    ctx_algo = ExecutionContext(
        prompt=prompt_algo,
        scope={
            "input_graph": PortSignature("input_graph", AlgebraicSignature("dict", "adjacency_dict")),
            "start_node": PortSignature("start_node", AlgebraicSignature("str", "source_node"))
        }
    )
    lines_algo = [UnificationGate.unify_cell(ctx_algo, c) for c in path_algo]
    code_algo = UnificationGate.resolve_imports("\n".join(lines_algo), ctx_algo)
    print("    Generated Code:\n" + "\n".join(f"      | {l}" for l in code_algo.splitlines()))

    # Assertions
    assert "dijkstra" in code_algo.lower(), "FAIL: dijkstra function missing in generated code!"
    assert "{'A': {'B': 1" not in code_algo, "FAIL: Hardcoded toy graph literal found in code!"

    # Execute in GEVR sandbox with dynamic graph fixture in prelude
    test_exec_code = (
        "input_graph = {'A': {'B': 1, 'C': 4}, 'B': {'A': 1, 'C': 2, 'D': 5}, 'C': {'A': 4, 'B': 2, 'D': 1}, 'D': {'B': 5, 'C': 1}}\n"
        "start_node = 'A'\n"
        + code_algo + "\n"
        "assert algorithm_out['D'] == 4, f'Expected dist 4 to D, got {algorithm_out.get(\"D\")}'\n"
    )

    verified_algo, stdout_algo, msg_algo = sandbox.execute_and_verify(test_exec_code)
    print(f"    Execution: verified={verified_algo}, msg='{(msg_algo or stdout_algo).strip()}'")

    if not verified_algo:
        print("    [!] Algorithmic Pipeline FAILED execution.")
        all_passed = False
    else:
        print("    [+] Algorithmic Pipeline PASSED and VERIFIED with shortest path dist=4!")

    # -------------------------------------------------------------
    # Test 4: Multi-Port Sort Pipeline (Sort by age ascending)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print(" [4] Multi-Port Pipeline: Read -> Sort Values -> Write")
    print("-" * 50)
    prompt_sort = "load input.csv and sort by age ascending then save to output.csv"
    print(f"    Prompt: '{prompt_sort}'")

    path_sort, _ = router.plan_path(prompt_sort)
    print(f"    Router Path: {[c.cell_id for c in path_sort]}")

    ctx_sort = ExecutionContext(prompt=prompt_sort)
    lines_sort = [UnificationGate.unify_cell(ctx_sort, c) for c in path_sort]
    code_sort = UnificationGate.resolve_imports("\n".join(lines_sort), ctx_sort)
    print("    Generated Code:\n" + "\n".join(f"      | {l}" for l in code_sort.splitlines()))

    assert "sort_values" in code_sort, "FAIL: sort_values missing in sort pipeline!"
    assert 'by="age"' in code_sort.replace("'", '"'), "FAIL: by='age' missing in sort_values!"
    assert "ascending=True" in code_sort, "FAIL: ascending=True missing in sort_values!"

    dummy_csv = os.path.join(ROOT_DIR, "input.csv")
    out_csv = os.path.join(ROOT_DIR, "output.csv")
    with open(dummy_csv, "w") as f:
        f.write("name,age\nAlice,40\nBob,25\nCharlie,35\n")

    verified_sort, stdout_sort, msg_sort = sandbox.execute_and_verify(code_sort)
    print(f"    Execution: verified={verified_sort}, msg='{(msg_sort or stdout_sort).strip()}'")

    if os.path.exists(out_csv):
        df_sorted = pd.read_csv(out_csv)
        ages = list(df_sorted["age"])
        assert ages == [25, 35, 40], f"FAIL: Ages not sorted ascending! Got: {ages}"
        os.remove(out_csv)
        print("    [+] Sort Verification: output.csv is properly sorted by age ascending [25, 35, 40]!")
    else:
        print("    [!] ERROR: output.csv was not created!")
        verified_sort = False

    if os.path.exists(dummy_csv):
        os.remove(dummy_csv)

    if not verified_sort:
        print("    [!] Multi-Port Sort Pipeline FAILED execution.")
        all_passed = False
    else:
        print("    [+] Multi-Port Sort Pipeline PASSED and VERIFIED cleanly!")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print(" [ALL 4 PHASE 0 HARDENED SMOKE TESTS PASSED & EXECUTED CLEANLY — 100%]")
    else:
        print(" [SOME TESTS FAILED — CHECK DETAILED LOGS ABOVE]")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

