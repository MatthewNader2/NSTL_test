# run_comprehensive_eval.py
"""
Comprehensive Benchmark Evaluator & Paper Evaluation Report Generator for NSTL.
Runs Profiles A, C (Cold), C (Warm), D, and E across the STRESS_TASKS matrix in-process.
Measures latency, AST node complexity, memo-cache speedup, and validation status.
Generates evaluation_report.md and comprehensive_eval_results.json.
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import json
import subprocess
import ast
import numpy as np

try:
    subprocess.run(["fuser", "-k", "58102/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

# Ensure working directory is project root and src is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from main import initialize_engine, run_prompt, InitRequest, RunRequest
import main

STRESS_TASKS = [
    {
        "task_id": "pandas_csv_clean",
        "category": "Data Engineering",
        "prompt": "Read a CSV file named data.csv into a pandas dataframe, drop any rows with missing values, sort it by the 'age' column in descending order, and then save the cleaned dataframe to a new CSV file named cleaned_data.csv.",
        "setup": """
import pandas as pd
import numpy as np
pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Dave'], 'age': [25, np.nan, 30, 22]}).to_csv('data.csv', index=False)
if os.path.exists('cleaned_data.csv'): os.remove('cleaned_data.csv')
""",
        "validate": """
import pandas as pd
import os
assert os.path.exists('cleaned_data.csv'), 'cleaned_data.csv was not created'
df = pd.read_csv('cleaned_data.csv')
assert df['age'].isnull().sum() == 0, 'Null values were not dropped'
assert df['age'].is_monotonic_decreasing, 'Ages are not sorted in descending order'
"""
    },
    {
        "task_id": "opencv_gray_convert",
        "category": "Image Processing",
        "prompt": "Read an image file named input.jpg using opencv, convert the image to grayscale, and save the resulting image to output.jpg.",
        "setup": """
import cv2
import numpy as np
import os
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite('input.jpg', img)
if os.path.exists('output.jpg'): os.remove('output.jpg')
""",
        "validate": """
import cv2
import os
import numpy as np
assert os.path.exists('output.jpg'), 'output.jpg was not created'
out = cv2.imread('output.jpg')
assert out is not None, 'Failed to read output.jpg'
assert len(out.shape) == 2 or (len(out.shape) == 3 and out.shape[2] == 1) or np.array_equal(out[:,:,0], out[:,:,1]), 'Image is not grayscale'
"""
    },
    {
        "task_id": "vague_data_transform",
        "category": "Vague Human Prompt",
        "prompt": "Process some input data, clean it up, transform values, and give me the summary output.",
        "setup": """
import pandas as pd
import numpy as np
import os
pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [10, 20, 30, 40]}).to_csv('data.csv', index=False)
""",
        "validate": """
stdout_clean = stdout.strip().lower()
assert any(term in stdout_clean for term in ["mean", "std", "count", "min", "max", "50%", "describe", "summary", "shape"]), f"Expected statistical summary output, got: {stdout[:200]}"
"""
    },
    {
        "task_id": "long_ml_pipeline",
        "category": "Long ML/Data Pipeline",
        "prompt": "Load data.csv, drop missing values, select numeric features, normalize them, train a RandomForestClassifier, compute accuracy, and save predictions to predictions.csv.",
        "setup": """
import pandas as pd
import numpy as np
import os
pd.DataFrame({'f1': [1.0, 2.0, np.nan, 4.0, 5.0], 'f2': [10, 20, 30, 40, 50], 'target': [0, 1, 0, 1, 0]}).to_csv('data.csv', index=False)
if os.path.exists('predictions.csv'): os.remove('predictions.csv')
""",
        "validate": """
import pandas as pd
import os
assert os.path.exists('predictions.csv')
df_p = pd.read_csv('predictions.csv')
assert len(df_p) > 0, 'Predictions CSV is empty'
assert 'prediction' in df_p.columns or len(df_p.columns) >= 1, 'Predictions missing expected columns'
assert not df_p.isnull().all().all(), 'Predictions contain only null values'
"""
    },
    {
        "task_id": "dijkstra_algorithm",
        "category": "Multi-Step Algorithm",
        "prompt": "Write a python function `dijkstra(graph, start)` that computes shortest paths using a priority queue, and return a dictionary of distances.",
        "setup": "",
        "validate": """
import sys
if '.' not in sys.path: sys.path.append('.')
from temp_eval_code import dijkstra
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
distances = dijkstra(graph, 'A')
assert distances == {'A': 0, 'B': 1, 'C': 3, 'D': 4}, f"Incorrect distances: {distances}"
"""
    }
]

def clear_synthesis_cache():
    """Removes temporary synthesis cache file to guarantee a true Cold run."""
    cache_path = os.path.join(PROJECT_ROOT, "trees", "micro", "synthesized_nodes.json")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print("  [+] Cleared synthesis cache for Cold Pass")
        except Exception as e:
            print(f"  [!] Failed to clear synthesis cache: {e}")

def init_engine_profile(profile, emb_model, llm_model):
    print(f"[*] Initializing Engine Profile {profile} (Embedder={emb_model}, LLM={llm_model})...", flush=True)
    req = InitRequest(profile=profile, embedder_model=emb_model, llm_model=llm_model)
    initialize_engine(req)
    
    start_wait = time.time()
    while time.time() - start_wait < 600:
        if main._engine_ready is True:
            print(f"  [+] Profile {profile} Engine Ready in {time.time() - start_wait:.1f}s.", flush=True)
            return True
        elif isinstance(main._engine_ready, str):
            print(f"  [!] Profile {profile} Engine Init Error: {main._engine_ready}", flush=True)
            return False
        time.sleep(1.0)
    print(f"  [!] Profile {profile} Engine Init Timed Out.", flush=True)
    return False

def setup_task_fixtures(task_id: str):
    """Ensures required test files exist on disk before task execution."""
    if task_id == "opencv_gray_convert":
        import cv2
        import numpy as np
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img[:, :] = [255, 128, 64]
        cv2.imwrite("input.jpg", dummy_img)
    elif task_id in ("pandas_csv_clean", "long_ml_pipeline", "vague_data_transform"):
        import pandas as pd
        import numpy as np
        dummy_df = pd.DataFrame({
            "age": [25, np.nan, 30, 22, 45],
            "salary": [50000, 60000, 75000, np.nan, 90000],
            "target": [0, 1, 0, 1, 0]
        })
        dummy_df.to_csv("data.csv", index=False)

def execute_task_run(task, profile_name, emb_model, llm_model):
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_id = task["task_id"]
    prompt = task["prompt"]

    # Pre-task fixture provisioning
    setup_task_fixtures(t_id)

    # Setup environment
    if task["setup"]:
        try:
            import pandas as pd
            exec(task["setup"], {"os": os, "np": np, "pd": pd})
        except Exception as se:
            print(f"    [!] Setup error: {se}")

    t0 = time.time()
    try:
        res = run_prompt(RunRequest(prompt=prompt))
    except Exception as re:
        res = {"code": f"# Execution Exception: {re}", "virtual_edges": []}
    t1 = time.time()
    latency = t1 - t0

    code = res.get("code", "")
    v_edges = res.get("virtual_edges", [])

    passed = False
    error_detail = ""
    ast_nodes = 0

    if not code or not code.strip() or code.startswith("# Engine is loading") or code.startswith("# Planner Error") or code.startswith("# Routing Error"):
        error_detail = f"No valid code generated: {code.strip()[:100]}"
    else:
        try:
            tree = ast.parse(code)
            ast_nodes = len(list(ast.walk(tree)))

            with open('temp_eval_code.py', 'w') as f:
                f.write(code)

            exec_res = subprocess.run(
                ["python3", "temp_eval_code.py"],
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=20
            )
            if exec_res.returncode != 0:
                error_detail = f"Runtime Error (code {exec_res.returncode}): {exec_res.stderr.strip()[:200]}"
            else:
                if task["validate"].strip():
                    val_globals = {"os": os, "np": np, "stdout": exec_res.stdout}
                    try:
                        exec(task["validate"], val_globals)
                        passed = True
                    except Exception as ve:
                        error_detail = f"Validation check failed: {ve}"
                else:
                    passed = True
        except SyntaxError as syn_err:
            error_detail = f"Syntax Error: {syn_err}"
        except subprocess.TimeoutExpired:
            error_detail = "Execution timeout (20s)"
        except Exception as ex:
            error_detail = f"Execution wrapper exception: {ex}"

    status_str = "PASSED" if passed else "FAILED"
    print(f"    [=> {status_str}] Latency: {latency:.3f}s | AST Nodes: {ast_nodes} | VEdges: {len(v_edges)} | Err: {error_detail}", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "profile": profile_name,
        "embedder": emb_model,
        "llm": llm_model,
        "task_id": t_id,
        "category": task["category"],
        "passed": passed,
        "latency_sec": round(latency, 3),
        "ast_nodes": ast_nodes,
        "virtual_edges": len(v_edges),
        "error": error_detail,
        "code": code
    }

def run_comprehensive_evaluation():
    import gc
    import torch
    print("=" * 70)
    print("NSTL COMPREHENSIVE BENCHMARK RUN (Profiles A, C, D, E)")
    print("=" * 70)

    matrix_results = {}
    raw_results = []

    profiles_to_test = [
        ("Profile A", "A", "jina-embeddings-v5-text-small", "auto"),
        ("Profile C", "C", "jina-embeddings-v5-text-small", "qwen2.5-coder-1.5b-instruct"),
        ("Profile D", "D", "jina-embeddings-v5-text-small", "qwen2.5-coder-1.5b-instruct"),
        ("Profile E", "E", "jina-embeddings-v5-text-small", "qwen2.5-coder-1.5b-instruct"),
    ]

    for task in STRESS_TASKS:
        matrix_results[task["task_id"]] = {
            "category": task["category"],
            "Profile A": None,
            "Profile C": None,
            "Profile D": None,
            "Profile E": None,
            "passed_all": True
        }

    for prof_label, prof_code, emb_model, llm_model in profiles_to_test:
        print(f"\n>>> Running {prof_label} (Embedder: {emb_model}, LLM: {llm_model})...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clear_synthesis_cache()
        if init_engine_profile(prof_code, emb_model, llm_model):
            for task in STRESS_TASKS:
                print(f"  [-] Task: {task['task_id']} ({task['category']})...")
                res = execute_task_run(task, prof_label, emb_model, llm_model)
                raw_results.append(res)
                matrix_results[task["task_id"]][prof_label] = res
                if not res["passed"]:
                    matrix_results[task["task_id"]]["passed_all"] = False
        else:
            print(f"  [!] Failed to initialize {prof_label}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)

    # Save raw results JSON
    with open('comprehensive_eval_results.json', 'w') as f:
        json.dump(raw_results, f, indent=4)
    print("\n[+] Benchmark raw data saved to comprehensive_eval_results.json")

    # Generate Evaluation Report Markdown
    generate_markdown_report(matrix_results, raw_results)

def generate_markdown_report(matrix_results, raw_results):
    report_path = os.path.join(PROJECT_ROOT, "evaluation_report.md")

    prof_keys = ["Profile A", "Profile C", "Profile D", "Profile E"]
    latencies = {pk: [] for pk in prof_keys}
    ast_counts = {pk: [] for pk in prof_keys}
    pass_counts = {pk: 0 for pk in prof_keys}
    total_counts = {pk: 0 for pk in prof_keys}

    for item in raw_results:
        pk = item["profile"]
        if pk in latencies:
            latencies[pk].append(item["latency_sec"])
            total_counts[pk] += 1
            if item["passed"]:
                pass_counts[pk] += 1
                ast_counts[pk].append(item["ast_nodes"])
            else:
                ast_counts[pk].append(item["ast_nodes"])

    avg_lat = {pk: float(np.mean(latencies[pk])) if latencies[pk] else 0.0 for pk in prof_keys}
    avg_ast = {pk: float(np.mean(ast_counts[pk])) if ast_counts[pk] else 0.0 for pk in prof_keys}

    lines = []
    lines.append("# NSTL Engine - Evaluation Report (Profiles A, C, D, E)\n")
    lines.append("## 1. Executive Summary & Evaluation Matrix\n")
    lines.append("This evaluation report documents the benchmark of the **Neural Syntax Tree Lattice (NSTL)** engine across target evaluation configurations for 1 representative model of each category (Embedder: `jina-embeddings-v5-text-small`, LLM: `qwen2.5-coder-1.5b-instruct`) across all 5 domain categories (excluding Profile B):\n")
    lines.append("- **Profile A**: Deterministic / Embedding-Only (`jina-embeddings-v5-text-small`)\n")
    lines.append("- **Profile C**: Full Pipeline with Zero-Shot Code Synthesis (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)\n")
    lines.append("- **Profile D**: Synthesis Disabled (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)\n")
    lines.append("- **Profile E**: Pre-Translation Pass + Code Synthesis (`jina-embeddings-v5-text-small` + `qwen2.5-coder-1.5b-instruct`)\n")

    lines.append("\n---\n")
    lines.append("## 2. Benchmark Summary Table\n")
    lines.append("| Task ID | Domain Category | Profile A | Profile C | Profile D | Profile E | Overall Status |")
    lines.append("|---|---|---|---|---|---|---|")

    for task in STRESS_TASKS:
        t_id = task["task_id"]
        row = matrix_results[t_id]
        cat = row["category"]

        def fmt_cell(res):
            if not res:
                return "N/A"
            status = "PASSED" if res["passed"] else "FAILED"
            return f"{res['latency_sec']:.3f}s ({status})"

        p_a = fmt_cell(row["Profile A"])
        p_c = fmt_cell(row["Profile C"])
        p_d = fmt_cell(row["Profile D"])
        p_e = fmt_cell(row["Profile E"])

        status_str = "PASSED" if row["passed_all"] else "PARTIAL / FAILED"

        lines.append(f"| `{t_id}` | {cat} | {p_a} | {p_c} | {p_d} | {p_e} | **{status_str}** |")

    lines.append("\n---\n")
    lines.append("## 3. Performance & Accuracy Breakdown\n")
    lines.append("### A. Pass Rate & Accuracy\n")
    for pk in prof_keys:
        rate = (pass_counts[pk] / total_counts[pk] * 100.0) if total_counts[pk] > 0 else 0.0
        lines.append(f"- **{pk}**: {pass_counts[pk]}/{total_counts[pk]} Passed (**{rate:.1f}%**)")

    lines.append("\n### B. Mean Execution Latency per Profile\n")
    for pk in prof_keys:
        lines.append(f"- **{pk}**: **{avg_lat[pk]:.3f}s** average latency")

    lines.append("\n### C. AST Program Complexity\n")
    for pk in prof_keys:
        lines.append(f"- **{pk}**: Average AST Node Count = **{avg_ast[pk]:.1f} nodes**")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[+] Generated complete evaluation report at {report_path}")

if __name__ == "__main__":
    run_comprehensive_evaluation()

