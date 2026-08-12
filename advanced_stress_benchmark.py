import sys
import os
import time
import urllib.request
import json
import subprocess

STRESS_TEST_CASES = [
    {
        "category": "Ultra-Vague Data Request",
        "prompt": "Process some input data, clean it up, transform values, and give me the summary output."
    },
    {
        "category": "Multi-Step Algorithm",
        "prompt": "Write a python function `dijkstra(graph, start)` that computes shortest paths using a priority queue, and return a dictionary of distances."
    },
    {
        "category": "Long ML/Data Pipeline",
        "prompt": "Load data.csv, drop missing values, select numeric features, normalize them, train a RandomForestClassifier, compute accuracy, and save predictions to predictions.csv."
    },
    {
        "category": "Edge Case & Type Coercion",
        "prompt": "Read a JSON string, convert dictionary keys into a pandas dataframe, transpose the dataframe, drop empty rows, and output to output.json."
    }
]

def start_server():
    env = os.environ.copy()
    env["TEST_HEADLESS"] = "1"
    proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd="/media/matthew/New Volume/grad_test/nstl_prototype")
    
    while True:
        try:
            urllib.request.urlopen("http://127.0.0.1:58102/api/status", timeout=1)
            break
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                raise RuntimeError("Server process died during startup")
    return proc

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
        print(f"Initialization failed: {e}")
        return False
        
    start_wait = time.time()
    while time.time() - start_wait < 120:
        try:
            resp = json.loads(urllib.request.urlopen("http://127.0.0.1:58102/api/status").read().decode())
            if resp.get("status") == "ready":
                return True
        except:
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
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        return resp.get("code", ""), resp.get("path", []), resp.get("virtual_edges", [])
    except Exception as e:
        print(f"Generation error: {e}")
        return None, [], []

if __name__ == "__main__":
    print("Advanced Stress Benchmark Module")
