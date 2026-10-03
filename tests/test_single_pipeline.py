"""
tests/test_single_pipeline.py - Neuro-Symbolic Topological Lattice (NSTL)
Consolidated Single-Prompt Multi-Profile / Multi-Method Evaluation Runner.

Tests the cross-domain tabular-to-signal processing prompt:
Original (User):
  "load a csv file, clean it , normalize X column, calculate the mean of Y column, then perfrom FFT on (X+Y) and make the results into Z"

Corrected (Canonical Evaluation):
  "Load dataset from 'data.csv', clean missing values, normalize column 'X', calculate the mean of column 'Y', then perform FFT on the sum of columns 'X' and 'Y', and store the results into Z"

Key Topological Challenges Tested:
1. Avoids the trivial lexical clue 'read', forcing semantic retrieval of 'pd.read_csv'.
2. Tabular-to-Numerical Carrier Bridge: Column operations (load, clean, normalize) exist in the
   tabular Pandas domain (DataFrame/Series). FFT exists in the numerical NumPy array domain (ndarray).
   The system must organically synthesize or discover the carrier transition ('to_numpy()')
   without any explicit mention of 'numpy' or 'to_numpy' in the prompt.
3. Multi-profile ablation: Evaluates Profile 0 (Symbolic baseline), Profile A (Embeddings only),
   and Profile C (Embedder + Local LLM with M7 Pathfinder & M9 Milestones).
"""

import os
import sys
import time
import ast
import json
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import warnings
warnings.filterwarnings("ignore")

from lattice import LatticeOrchestrator, Cell
from unification import UnificationGate, ExecutionContext, TypeRegistry
from inference import ModelManager

# Define prompts
RAW_USER_PROMPT = "load a csv file, clean it , normalize X column, calculate the mean of Y column, then perfrom FFT on (X+Y) and make the results into Z"
CORRECTED_PROMPT = "Load dataset from 'data.csv', clean missing values, normalize column 'X', calculate the mean of column 'Y', then perform FFT on the sum of columns 'X' and 'Y', and store the results into Z"

# Selected representative configurations: (Profile, Method, Embedder, LLM, Description)
EVAL_CONFIGURATIONS = [
    {
        "profile": "0",
        "method": "m0",
        "embedder": None,
        "llm": None,
        "name": "Profile 0 (Pure Symbolic / M0 Trellis)",
        "desc": "Baseline algebraic type propagation without neural embedder or LLM"
    },
    {
        "profile": "A",
        "method": "m1",
        "embedder": "jina-embeddings-v5-text-nano",
        "llm": None,
        "name": "Profile A (Dense Embeddings / M1 Clause Anchor)",
        "desc": "Vector semantic retrieval + Robinson unification anchors (no LLM)"
    },
    {
        "profile": "A",
        "method": "m6",
        "embedder": "jina-embeddings-v5-text-nano",
        "llm": None,
        "name": "Profile A (Dense Embeddings / M6 Hybrid Anchors)",
        "desc": "Flagship hybrid anchor combining BM25, dense cosine, and subgraphs"
    },
    {
        "profile": "C",
        "method": "m7",
        "embedder": "jina-embeddings-v5-text-nano",
        "llm": "qwen2.5-coder-1.5b-instruct",
        "name": "Profile C (Embedder + LLM / M7 Pathfinder)",
        "desc": "Global LLM path proposal verified by Robinson unification with auto-bridging"
    },
    {
        "profile": "C",
        "method": "m9",
        "embedder": "jina-embeddings-v5-text-nano",
        "llm": "qwen2.5-coder-1.5b-instruct",
        "name": "Profile C (Embedder + LLM / M9 Milestones)",
        "desc": "Essential milestone identification with multi-hop topological bridge synthesis"
    }
]


def ensure_dummy_data():
    """Ensure a realistic 'data.csv' exists for live dry-run execution."""
    data_path = os.path.join(ROOT_DIR, "data.csv")
    if not os.path.exists(data_path):
        import numpy as np
        import pandas as pd
        t = np.linspace(0, 1, 100)
        df = pd.DataFrame({
            "X": np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(100),
            "Y": np.cos(2 * np.pi * 5 * t) + 0.1 * np.random.randn(100),
            "Z": np.zeros(100)
        })
        # Add a missing value to test cleaning
        df.iloc[5, 0] = np.nan
        df.to_csv(data_path, index=False)
        print(f"[TEST SETUP] Created sample dataset '{data_path}' with missing value.")


def run_configuration(cfg: Dict[str, Any], prompt: str, orchestrator: LatticeOrchestrator) -> Dict[str, Any]:
    prof_name = cfg["profile"]
    m_name = cfg["method"]
    emb_model = cfg["embedder"]
    llm_model = cfg["llm"]

    print(f"\n{'='*75}")
    print(f" RUNNING: {cfg['name']}")
    print(f" Description: {cfg['desc']}")
    print(f"{'='*75}")

    t_start = time.time()
    mm = ModelManager.get_instance()

    # 1. Profile Initialization
    try:
        if prof_name == "0":
            mm.cleanup()
            rag = None
        else:
            mm.initialize_profile(prof_name, embedder_name=emb_model or "", llm_name=llm_model or "")
            from internal_rag import LocalRAG
            rag = LocalRAG(trees_dir=os.path.join(ROOT_DIR, "trees"), orchestrator=orchestrator)
    except Exception as e:
        print(f"[-] Profile initialization failed: {e}")
        return {
            "config": cfg["name"],
            "profile": prof_name,
            "method": m_name,
            "path": [],
            "has_bridge": False,
            "code": "",
            "valid_syntax": False,
            "exec_success": False,
            "elapsed_sec": round(time.time() - t_start, 2),
            "error": f"Init error: {e}"
        }

    # 2. Router & Planning
    from router import LatticeRouter
    router = LatticeRouter(orchestrator=orchestrator, internal_rag=rag, default_route_method=m_name)
    gate = UnificationGate(orchestrator=orchestrator)

    path: List[Cell] = []
    error_msg = ""
    try:
        path, _ = router.plan_path(prompt, route_method=m_name)
        cell_ids = [c.cell_id for c in path]
        print(f"  [+] Planned Path ({len(path)} steps): {' -> '.join(cell_ids)}")
    except Exception as e:
        error_msg = f"Planning error: {e}"
        print(f"  [-] Planning failed: {e}")

    # Check for carrier bridge
    has_bridge = any(
        ("NUMPY" in c.cell_id and "PANDAS" in c.cell_id) or
        ("TO_NUMPY" in c.cell_id) or
        ("PROJECT" in c.cell_id and "NUMPY" in c.cell_id)
        for c in path
    )
    if has_bridge:
        bridge_nodes = [c.cell_id for c in path if "NUMPY" in c.cell_id and ("PANDAS" in c.cell_id or "TO_NUMPY" in c.cell_id or "PROJECT" in c.cell_id)]
        print(f"  [+] Carrier Bridge Detected: {bridge_nodes} (organic tabular -> array transition)")

    # 3. Unification & Code Emission
    code = ""
    valid_syntax = False
    exec_success = False

    if path:
        try:
            ctx = ExecutionContext(prompt=prompt)
            code = gate.emit_code(path, ctx)
            print(f"\n  --- Emitted Python Code ---")
            for line in code.strip().split("\n"):
                print(f"  | {line}")
            print(f"  ---------------------------\n")

            # Syntax verification via AST
            ast.parse(code)
            valid_syntax = True
            print(f"  [+] Code Syntax Verification: PASSED (AST valid)")

            # Live execution test (Sandbox/Isolated scope)
            exec_scope = {}
            # Run in cwd so data.csv is accessible
            old_cwd = os.getcwd()
            os.chdir(ROOT_DIR)
            try:
                exec(code, exec_scope)
                exec_success = True
                print(f"  [+] Live Execution Test: PASSED (Code executed without runtime errors)")
                # Inspect final scope variables
                result_vars = {k: type(v).__name__ for k, v in exec_scope.items() if not k.startswith("__") and k not in ("pd", "np", "plt", "cv2")}
                print(f"  [+] Execution Scope Output Variables: {result_vars}")
            except Exception as ex:
                exec_success = False
                print(f"  [-] Live Execution Test: FAILED ({ex})")
            finally:
                os.chdir(old_cwd)

        except Exception as e:
            error_msg = error_msg or f"Emission/unification error: {e}"
            print(f"  [-] Code emission / unification failed: {e}")

    elapsed = round(time.time() - t_start, 2)
    print(f"  [+] Elapsed Time: {elapsed}s")

    return {
        "config": cfg["name"],
        "profile": prof_name,
        "method": m_name,
        "path": [c.cell_id for c in path],
        "has_bridge": has_bridge,
        "code": code,
        "valid_syntax": valid_syntax,
        "exec_success": exec_success,
        "elapsed_sec": elapsed,
        "error": error_msg
    }


def main():
    print(f"\n===============================================================================")
    print(f" NSTL SINGLE-PROMPT PIPELINE EVALUATION SUITE")
    print(f"===============================================================================")
    print(f"Raw User Prompt:\n  \"{RAW_USER_PROMPT}\"\n")
    print(f"Corrected Canonical Prompt:\n  \"{CORRECTED_PROMPT}\"\n")
    print(f"Rationale for Prompt Formulation:")
    print(f"  1. Replaced 'read' with 'Load' so lexical search cannot trivially match 'pd.read_csv'.")
    print(f"  2. Bound input file to 'data.csv' to allow concrete variable binding and live execution.")
    print(f"  3. Corrected typos ('perfrom' -> 'perform', clarified '(X+Y)' to sum of columns).")
    print(f"  4. Maintained ZERO mention of numpy, ndarray, or .to_numpy(), testing if the type")
    print(f"     algebra autonomously bridges DataFrame -> ndarray for the FFT transformation.")
    print(f"===============================================================================\n")

    ensure_dummy_data()

    trees_dir = os.path.join(ROOT_DIR, "trees")
    orchestrator = LatticeOrchestrator(trees_directory=trees_dir)
    print(f"[LATTICE] Loaded {len(orchestrator.loaded_cells)} micro-cells from trees.")

    results: List[Dict[str, Any]] = []

    for cfg in EVAL_CONFIGURATIONS:
        res = run_configuration(cfg, CORRECTED_PROMPT, orchestrator)
        results.append(res)

    # Print summary table
    print(f"\n\n{'='*95}")
    print(f" EVALUATION SUMMARY RESULTS")
    print(f"{'='*95}")
    header = f"{'Configuration':<42} | {'Steps':<5} | {'Bridge?':<7} | {'Syntax':<7} | {'Exec':<7} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        bridge_str = "YES" if r["has_bridge"] else "NO"
        syntax_str = "PASS" if r["valid_syntax"] else "FAIL"
        exec_str = "PASS" if r["exec_success"] else "FAIL"
        steps_str = str(len(r["path"]))
        time_str = f"{r['elapsed_sec']:.2f}"
        print(f"{r['config']:<42} | {steps_str:<5} | {bridge_str:<7} | {syntax_str:<7} | {exec_str:<7} | {time_str:<8}")
    print(f"{'='*95}\n")

    # Clean up model manager at end
    ModelManager.get_instance().cleanup()
    print("[TEST SUITE] Completed cleanly.")


if __name__ == "__main__":
    main()
