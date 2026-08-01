import sys
import os
import time
import urllib.request
import json
import threading
import subprocess
import traceback

def start_server():
    env = os.environ.copy()
    env["TEST_HEADLESS"] = "1"
    # Ensure it's using the correct python from environment if needed, but 'python3' is fine
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
    print("Sending initialize...")
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
    while time.time() - start_wait < 1200:
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

def run_eval():
    with open('eval_dataset.json', 'r') as f:
        dataset = json.load(f)
        
    results = []

    EMBEDDINGS = ["jina-embeddings-v5-text-nano", "embeddinggemma-300m"]
    LLMS = ["qwen2.5-coder-0.5b-instruct", "qwen2.5-coder-1.5b-instruct", "Qwen2.5-Coder-7B-Instruct-GGUF"]
    
    MATRIX = []
    for emb in EMBEDDINGS:
        MATRIX.append(("A", emb, "auto"))
        for llm in LLMS:
             MATRIX.append(("C", emb, llm))
             MATRIX.append(("D", emb, llm))

    for profile, emb, llm in MATRIX:
        print(f"\n======================================")
        print(f"TESTING PROFILE {profile} (LLM: {llm}, EMB: {emb})")
        print(f"======================================")
        
        proc = start_server()
        ready = init_server(profile, emb, llm)
        
        if not ready:
            print("Failed to initialize server.")
            proc.terminate()
            proc.wait()
            continue
            
        for task in dataset:
            print(f"Running task: {task['task_id']}")
            
            # Setup
            if task.get('setup_script'):
                try:
                    exec(task['setup_script'], globals(), globals())
                except Exception as e:
                    print(f"Setup script failed: {e}")
                    
            # Generate
            code = generate_code(task['prompt'])
            
            passed = False
            error_msg = ""
            stdout_str = ""
            stderr_str = ""
            
            if code is None or not code.strip():
                error_msg = "No code generated."
            else:
                # Write to temp file
                with open('temp_eval_run.py', 'w') as f:
                    f.write(code)
                
                # Execute
                try:
                    run_proc = subprocess.run(["python3", "temp_eval_run.py"], capture_output=True, text=True, timeout=30)
                    stdout_str = run_proc.stdout
                    stderr_str = run_proc.stderr
                    
                    if run_proc.returncode != 0:
                        error_msg = f"Execution failed with return code {run_proc.returncode}:\n{stderr_str}"
                    else:
                        # Validate
                        validation_script = task.get('validation_script')
                        if validation_script:
                            # Pass stdout and stderr to the validation env
                            val_env = {"stdout": stdout_str, "stderr": stderr_str}
                            try:
                                exec(validation_script, val_env)
                                passed = True
                            except AssertionError as ae:
                                error_msg = f"Validation failed: {ae}"
                            except Exception as e:
                                error_msg = f"Validation script crashed: {e}\n{traceback.format_exc()}"
                        else:
                            passed = True # No validation script = pass if it didn't crash
                except subprocess.TimeoutExpired:
                    error_msg = "Execution timed out after 30 seconds."
                except Exception as e:
                    error_msg = f"Failed to run code: {e}"
            
            result = {
                "profile": profile,
                "embedder": emb,
                "llm": llm,
                "task_id": task["task_id"],
                "passed": passed,
                "error": error_msg,
                "stdout": stdout_str,
                "code": code
            }
            results.append(result)
            print(f"Task {task['task_id']} Passed: {passed}")
            if not passed:
                print(f"Error: {error_msg}")
        
        print("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluations completed. Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_eval()
