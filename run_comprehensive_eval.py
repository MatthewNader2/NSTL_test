"""
run_comprehensive_eval.py - NSTL Comprehensive Benchmark Runner
FIXED: Properly awaits async engine functions, uses spawn for subprocess safety,
       disables tokenizer parallelism before import, adds debug logging for empty code,
       and CRITICALLY polls for engine readiness before running tasks.
"""

import os
import sys
import json
import time
import ast
import tempfile
import shutil
import subprocess
import asyncio
import inspect
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method("spawn", force=True)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import initialize_engine, run_prompt, InitRequest, RunRequest
from log_config import get_logger

logger = get_logger("eval")

STRESS_TASKS: List[Dict[str, Any]] = [
    {
        "id": "pandas_csv_clean",
        "name": "pandas_csv_clean (Data Engineering)",
        "prompt": "Read data.csv, drop rows with missing values, sort by the 'age' column descending, and save to output.csv",
        "setup": """
import pandas as pd
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', None], 'age': [25, 30, 35, 28]})
df.to_csv('data.csv', index=False)
""",
        "validate": """
import pandas as pd, os
assert os.path.exists('output.csv'), "output.csv not found"
out = pd.read_csv('output.csv')
assert out['age'].is_monotonic_decreasing, "Not sorted descending by age"
assert out['age'].notna().all(), "Nulls not dropped"
""",
    },
    {
        "id": "opencv_gray_convert",
        "name": "opencv_gray_convert (Image Processing)",
        "prompt": "Load input.jpg, convert it to grayscale, and save as output.jpg",
        "setup": """
import numpy as np, cv2
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
cv2.imwrite('input.jpg', img)
""",
        "validate": """
import cv2, os
assert os.path.exists('output.jpg'), "output.jpg not found"
img = cv2.imread('output.jpg')
assert img is not None, "Could not read output.jpg"
""",
    },
    {
        "id": "vague_data_transform",
        "name": "vague_data_transform (Vague Human Prompt)",
        "prompt": "Clean the dataset and show summary statistics",
        "setup": """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, None, 4], 'b': [5, None, 7, 8]})
df.to_csv('data.csv', index=False)
""",
        "validate": "pass",
    },
    {
        "id": "long_ml_pipeline",
        "name": "long_ml_pipeline (Long ML/Data Pipeline)",
        "prompt": "Load data.csv, drop missing values, convert all columns to numeric, normalize features, train a RandomForestClassifier, and print accuracy",
        "setup": """
import pandas as pd, numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'feat1': np.random.rand(50),
    'feat2': np.random.rand(50),
    'target': np.random.randint(0, 2, 50)
})
df.to_csv('data.csv', index=False)
""",
        "validate": "pass",
    },
    {
        "id": "dijkstra_algorithm",
        "name": "dijkstra_algorithm (Multi-Step Algorithm)",
        "prompt": "Implement Dijkstra's shortest path algorithm on a sample graph and print the distances",
        "setup": "pass",
        "validate": "pass",
    },
]


async def run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["id"]
    prompt = task["prompt"]

    tmpdir = tempfile.mkdtemp(prefix=f"nstl_eval_{task_id}_")
    orig_cwd = os.getcwd()
    os.chdir(tmpdir)

    try:
        setup_code = task.get("setup", "")
        if setup_code and setup_code.strip() and setup_code.strip() != "pass":
            setup_globals = {
                "__builtins__": __builtins__,
                "os": __import__("os"),
                "sys": __import__("sys"),
                "pd": __import__("pandas"),
                "np": __import__("numpy"),
                "cv2": __import__("cv2"),
            }
            exec(setup_code, setup_globals)

        start_time = time.time()
        try:
            req = RunRequest(prompt=prompt)
            raw_result = run_prompt(req)
            if inspect.isawaitable(raw_result):
                result = await raw_result
            else:
                result = raw_result

            # Small async yield to let background threads breathe and finalize state
            await asyncio.sleep(0.1)

            latency = time.time() - start_time
        except Exception as e:
            logger.error(f"[EVAL] Engine error on {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "ENGINE_ERROR",
                "latency": time.time() - start_time,
                "error": str(e),
            }

        generated_code = result.get("code", "") if isinstance(result, dict) else ""
        if not generated_code:
            generated_code = result.get("generated_code", "") if isinstance(result, dict) else ""

        if not generated_code:
            print(f"    [DEBUG] Empty code. Full engine response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            print(f"    [DEBUG] Status: {result.get('status') if isinstance(result, dict) else 'N/A'}")
            print(f"    [DEBUG] Message: {result.get('message', 'N/A') if isinstance(result, dict) else 'N/A'}")

        code_path = os.path.join(tmpdir, "temp_eval_code.py")
        with open(code_path, "w") as f:
            f.write(generated_code or "pass")

        exec_status = "UNKNOWN"
        exec_error = None
        try:
            proc = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=tmpdir,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}
            )
            if proc.returncode == 0:
                exec_status = "PASSED"
            else:
                exec_status = "RUNTIME_ERROR"
                exec_error = proc.stderr[-800:] if proc.stderr else "No stderr"
        except subprocess.TimeoutExpired:
            exec_status = "TIMEOUT"
        except Exception as e:
            exec_status = "EXEC_EXCEPTION"
            exec_error = str(e)

        validation_status = "SKIPPED"
        validation_error = None
        validate_code = task.get("validate", "")
        if validate_code and validate_code.strip() and validate_code.strip() != "pass" and exec_status == "PASSED":
            try:
                val_globals = {
                    "__builtins__": __builtins__,
                    "os": __import__("os"),
                    "pd": __import__("pandas"),
                    "np": __import__("numpy"),
                }
                exec(validate_code, val_globals)
                validation_status = "PASSED"
            except Exception as e:
                validation_status = "FAILED"
                validation_error = str(e)

        try:
            tree = ast.parse(generated_code or "pass")
            ast_nodes = len(list(ast.walk(tree)))
        except Exception:
            ast_nodes = 0

        status = "PASSED" if (exec_status == "PASSED" and validation_status in ("PASSED", "SKIPPED")) else "FAILED"

        return {
            "task_id": task_id,
            "status": status,
            "latency": round(latency, 3),
            "ast_nodes": ast_nodes,
            "exec_status": exec_status,
            "exec_error": exec_error,
            "validation_status": validation_status,
            "validation_error": validation_error,
            "generated_code": generated_code,
        }

    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


async def run_profile(profile: Dict[str, Any], all_results: List[Dict]):
    print(f"\n{'='*70}")
    print(f">>> Running Profile {profile['name']} (Embedder: {profile['embedder_model']}, LLM: {profile['llm_model'] or 'auto'})...")
    print(f"{'='*70}")

    t0 = time.time()
    init_req = InitRequest(
        profile=profile["name"],
        embedder_model=profile["embedder_model"],
        llm_model=profile["llm_model"],
        embedder_device="auto",
        llm_device="auto",
        trees_storage="ram"
    )
    init_result = initialize_engine(init_req)
    print(f"  [INIT] {init_result.get('status')} on {init_result.get('device', 'unknown')}")

    # CRITICAL: Wait for background initialization thread to finish
    poll_start = time.time()
    while True:
        main_mod = sys.modules.get("main")
        if main_mod and getattr(main_mod, "_engine_ready", False):
            break
        await asyncio.sleep(0.5)
        if time.time() - poll_start > 300:
            print("  [INIT] TIMEOUT waiting for engine readiness")
            return

    init_time = time.time() - t0
    print(f"  [+] Profile {profile['name']} Engine Ready in {init_time:.1f}s.")

    for task in STRESS_TASKS:
        print(f"\n  [{'+' if task['id'] == 'pandas_csv_clean' else '-'}] Task: {task['name']}...")
        result = await run_single_task(task)
        all_results.append({"profile": profile["name"], **result})

        status_icon = "[=> PASSED]" if result["status"] == "PASSED" else "[=> FAILED]"
        err_snippet = (result.get("exec_error") or "None")[:80].replace("\n", " ")
        print(f"    {status_icon} Latency: {result['latency']:.3f}s | AST Nodes: {result['ast_nodes']} | Err: {err_snippet}")


async def main_async():
    profiles = [
        {"name": "A", "embedder_model": "jina-embeddings-v5-text-small", "llm_model": ""},
        {"name": "C", "embedder_model": "jina-embeddings-v5-text-small", "llm_model": "qwen2.5-coder-1.5b-instruct"},
        {"name": "D", "embedder_model": "jina-embeddings-v5-text-small", "llm_model": "qwen2.5-coder-1.5b-instruct"},
        {"name": "E", "embedder_model": "jina-embeddings-v5-text-small", "llm_model": "qwen2.5-coder-1.5b-instruct"},
    ]

    all_results: List[Dict] = []

    for profile in profiles:
        await run_profile(profile, all_results)

    results_path = PROJECT_ROOT / "comprehensive_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    report_path = PROJECT_ROOT / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# NSTL Comprehensive Evaluation Report\n\n")
        for profile_name in sorted({r["profile"] for r in all_results}):
            f.write(f"## Profile {profile_name}\n\n")
            profile_results = [r for r in all_results if r["profile"] == profile_name]
            passed = sum(1 for r in profile_results if r["status"] == "PASSED")
            total = len(profile_results)
            f.write(f"**Pass Rate:** {passed}/{total} ({100*passed/total:.1f}%)\n\n")
            f.write("| Task | Status | Latency | AST Nodes | Error |\n")
            f.write("|------|--------|---------|-----------|-------|\n")
            for r in profile_results:
                err = (r.get("exec_error") or "")[:60].replace("\n", " ")
                f.write(f"| {r['task_id']} | {r['status']} | {r['latency']:.2f}s | {r['ast_nodes']} | {err} |\n")
            f.write("\n")

    print(f"\n[+] Benchmark raw data saved to {results_path}")
    print(f"[+] Generated complete evaluation report at {report_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
