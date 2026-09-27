import sys
import os
import time
import urllib.request
import json
import subprocess

def start_server():
    subprocess.run("fuser -k 58102/tcp || pkill -9 -f 'python3 src/main.py' || true", shell=True, capture_output=True)
    time.sleep(1)
    env = os.environ.copy()
    env["TEST_HEADLESS"] = "1"
    proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd="/media/matthew/New Volume/grad_test/nstl_prototype")
    
    print("Waiting for server to boot...")
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
    print(f"Sending initialize: Profile={profile}, EMB={emb_model}, LLM={llm_model}...")
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
        return resp.get("code", "")
    except Exception as e:
        print(f"Generation error: {e}")
        return None

def test_failed():
    with open('eval_dataset.json', 'r') as f:
        dataset = json.load(f)
    dataset_dict = {t['task_id']: t for t in dataset}

    FAILED_CASES = [
        ("C", "jina-embeddings-v5-text-nano", "qwen2.5-coder-1.5b-instruct", "pandas_csv_clean"),
        ("C", "jina-embeddings-v5-text-nano", "qwen2.5-coder-1.5b-instruct", "math_add_function"),
        ("D", "jina-embeddings-v5-text-nano", "qwen2.5-coder-1.5b-instruct", "pandas_csv_clean"),
        ("C", "embeddinggemma-300m", "qwen2.5-coder-1.5b-instruct", "pandas_csv_clean"),
        ("D", "embeddinggemma-300m", "qwen2.5-coder-1.5b-instruct", "pandas_csv_clean"),
        ("D", "embeddinggemma-300m", "Qwen2.5-Coder-7B-Instruct-GGUF", "pandas_csv_clean"),
    ]

    results = []

    for profile, emb, llm, task_id in FAILED_CASES:
        print(f"\n======================================")
        print(f"RE-TESTING FAILED CASE: Profile {profile} | LLM: {llm} | EMB: {emb} | Task: {task_id}")
        print(f"======================================")
        
        proc = start_server()
        ready = init_server(profile, emb, llm)
        
        if not ready:
            print("Failed to initialize server.")
            proc.terminate()
            proc.wait()
            continue
            
        task = dataset_dict[task_id]
        if task.get('setup_script'):
            subprocess.run([sys.executable, "-c", task['setup_script']], check=True)
            
        code = generate_code(task['prompt'])
        passed = False
        error_msg = ""
        stdout_str = ""
        
        if code:
            with open('temp_eval_run.py', 'w') as tf:
                tf.write(code)
            
            run_res = subprocess.run([sys.executable, 'temp_eval_run.py'], capture_output=True, text=True)
            stdout_str = run_res.stdout
            if run_res.returncode != 0:
                error_msg = f"Execution failed with return code {run_res.returncode}:\n{run_res.stderr}"
            else:
                if task.get('verify_script'):
                    v_res = subprocess.run(task['verify_script'], shell=True, capture_output=True, text=True)
                    if v_res.returncode == 0:
                        passed = True
                    else:
                        error_msg = f"Validation failed: {v_res.stderr.strip() or v_res.stdout.strip()}"
                else:
                    passed = True
        else:
            error_msg = "Code generation returned empty code."
            
        if task.get('cleanup_script'):
            subprocess.run(task['cleanup_script'], shell=True)
            
        print(f"RESULT: Task {task_id} Passed = {passed}")
        if not passed:
            print(f"Error: {error_msg}")
        results.append({"profile": profile, "emb": emb, "llm": llm, "task": task_id, "passed": passed, "code": code})

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run("pkill -9 -f 'python3 src/main.py' || true", shell=True, capture_output=True)
        time.sleep(2)

    print("\n======================================")
    print("SUMMARY OF RE-TESTED FAILURE CASES:")
    for r in results:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f"Profile {r['profile']} | LLM: {r['llm']} | EMB: {r['emb']} | Task: {r['task']} => {status}")

if __name__ == "__main__":
    test_failed()
