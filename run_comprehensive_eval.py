import sys
import os
import time
import urllib.request
import json
import subprocess
import traceback
import ast
import numpy as np

# Ensure working directory is project root
PROJECT_ROOT = "/media/matthew/New Volume/grad_test/nstl_prototype"
os.chdir(PROJECT_ROOT)

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
assert os.path.exists('output.jpg'), 'output.jpg was not created'
out = cv2.imread('output.jpg')
assert out is not None, 'Failed to read output.jpg'
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
# Check that code parses and runs without fatal exceptions
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
# Check execution finish
"""
    },
    {
        "task_id": "dijkstra_algorithm",
        "category": "Multi-Step Algorithm",
        "prompt": "Write a python function `dijkstra(graph, start)` that computes shortest paths using a priority queue, and return a dictionary of distances.",
        "setup": "",
        "validate": """
# Check syntax & execution
"""
    }
]

PROFILES_TO_TEST = [
    ("A", "jina-embeddings-v5-text-nano", "auto"),
    ("C", "jina-embeddings-v5-text-small", "Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("D", "jina-embeddings-v5-text-small", "Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("E", "jina-embeddings-v5-text-small", "Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("E", "Qwen3-Embedding-0.6B-GGUF", "Qwen2.5-Coder-7B-Instruct-GGUF")
]

def start_server():
    subprocess.run("fuser -k 58102/tcp || pkill -9 -f 'python3 src/main.py' || true", shell=True, capture_output=True)
    time.sleep(1)
    env = os.environ.copy()
    env["TEST_HEADLESS"] = "1"
    proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd=PROJECT_ROOT)
    
    start_time = time.time()
    while time.time() - start_time < 45:
        try:
            urllib.request.urlopen("http://127.0.0.1:58102/api/status", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                # If server exited immediately due to port race condition, retry after 2s delay
                time.sleep(2)
                proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd=PROJECT_ROOT)
    raise RuntimeError("Server boot timed out")

def init_server(profile, emb_model, llm_model):
    init_payload = {"profile": profile, "embedder_model": emb_model, "llm_model": llm_model}
    req = urllib.request.Request(
        "http://127.0.0.1:58102/api/initialize", 
        data=json.dumps(init_payload).encode(), 
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Initialization API error: {e}")
        return False
        
    start_wait = time.time()
    while time.time() - start_wait < 600:
        try:
            resp = json.loads(urllib.request.urlopen("http://127.0.0.1:58102/api/status").read().decode())
            if resp.get("status") == "ready":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def generate_code(prompt):
    run_payload = {"prompt": prompt}
    req = urllib.request.Request(
        "http://127.0.0.1:58102/api/run", 
        data=json.dumps(run_payload).encode(), 
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
        t1 = time.time()
        return resp.get("code", ""), resp.get("path", []), resp.get("virtual_edges", []), (t1 - t0)
    except Exception as e:
        t1 = time.time()
        print(f"Generation error: {e}")
        return None, [], [], (t1 - t0)

def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run("pkill -9 -f 'python3 src/main.py' || true", shell=True, capture_output=True)
    time.sleep(2)

def run_eval():
    all_results = []
    
    print("=" * 60)
    print("STARTING NSTL COMPREHENSIVE BENCHMARK RUN Across Profiles A, C, D")
    print("=" * 60)

    for profile, emb, llm in PROFILES_TO_TEST:
        print(f"\n>>> Running Profile {profile} | Embedder: {emb} | LLM: {llm}")
        
        stop_server(None) # ensure clean port
        proc = start_server()
        ready = init_server(profile, emb, llm)
        
        if not ready:
            print(f"Failed to initialize Profile {profile} server.")
            stop_server(proc)
            continue

        for task in STRESS_TASKS:
            t_id = task["task_id"]
            prompt = task["prompt"]
            print(f"  [-] Task: {t_id}...")
            
            # Setup
            if task["setup"]:
                try:
                    exec(task["setup"], {"os": os, "np": np})
                except Exception as se:
                    print(f"    [!] Setup error: {se}")

            code, path, v_edges, latency = generate_code(prompt)
            
            passed = False
            error_detail = ""
            ast_nodes = 0
            
            if not code or not code.strip():
                error_detail = "No code generated / Timeout"
            else:
                try:
                    tree = ast.parse(code)
                    ast_nodes = len(list(ast.walk(tree)))
                    
                    # Execute generated code to verify runtime validity
                    with open('temp_eval_code.py', 'w') as f:
                        f.write(code)
                    
                    exec_res = subprocess.run(["python3", "temp_eval_code.py"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=20)
                    if exec_res.returncode != 0:
                        error_detail = f"Runtime Error (code {exec_res.returncode}): {exec_res.stderr.strip()[:200]}"
                    else:
                        # Validation script
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
            
            res_entry = {
                "profile": profile,
                "embedder": emb,
                "llm": llm,
                "task_id": t_id,
                "category": task["category"],
                "passed": passed,
                "latency_sec": round(latency, 3),
                "ast_nodes": ast_nodes,
                "virtual_edges": len(v_edges),
                "error": error_detail,
                "code": code
            }
            all_results.append(res_entry)
            status_str = "PASSED" if passed else "FAILED"
            print(f"    [=> {status_str}] Latency: {latency:.2f}s | AST Nodes: {ast_nodes} | VEdges: {len(v_edges)} | Err: {error_detail}")

        stop_server(proc)

    with open('comprehensive_eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE. Saved to comprehensive_eval_results.json")
    print("=" * 60)

if __name__ == "__main__":
    run_eval()
