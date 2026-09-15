import sys
import os
import time
import urllib.request
import json
import threading
import subprocess
import traceback
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
PROJECT_ROOT = config.PROJECT_ROOT
API_HOST = config.API_HOST
API_PORT = config.API_PORT

def start_server():
    env = os.environ.copy()
    env["TEST_HEADLESS"] = "1"
    # Ensure it's using the correct python from environment if needed, but 'python3' is fine
    proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd=str(PROJECT_ROOT))
    
    print("Waiting for server to boot...")
    while True:
        try:
            urllib.request.urlopen(f"http://{API_HOST}:{API_PORT}/api/status", timeout=1)
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
        f"http://{API_HOST}:{API_PORT}/api/initialize", 
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
            resp = json.loads(urllib.request.urlopen(f"http://{API_HOST}:{API_PORT}/api/status").read().decode())
            if resp.get("status") == "ready":
                return True
        except Exception as e:
            print(f"Status check failed: {e}")
        time.sleep(1)
    return False

def generate_code(prompt):
    run_payload = {"prompt": prompt}
    req = urllib.request.Request(
        f"http://{API_HOST}:{API_PORT}/api/run", 
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
    with open(os.path.join(str(PROJECT_ROOT), 'eval_dataset.json'), 'r') as f:
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
                    exec(task['setup_script'], {"__builtins__": __builtins__}, {})
                except Exception as e:
                    print(f"Setup script failed: {e}")
                    
            # Generate
            code = generate_code(task['prompt'])
            
            # Preflight static lint check (T1.1)
            is_type_safe = False
            preflight_violations = []
            try:
                from preflight import PreflightLinter
                # Basic AST parsing and structural check
                lint_res = PreflightLinter.lint([], prompt=task['prompt'], code_str=code)
                is_type_safe = lint_res.is_valid
                preflight_violations = lint_res.violations
            except Exception:
                try:
                    ast.parse(code or "")
                    is_type_safe = True
                except Exception as e:
                    preflight_violations.append(str(e))

            tier = "failed"
            passed = False
            error_msg = ""
            stdout_str = ""
            stderr_str = ""
            repaired_code = None
            pass_attribution = "none"

            if code is None or not code.strip():
                error_msg = "No code generated."
            else:
                # Write to temp file
                temp_file = os.path.join(str(PROJECT_ROOT), 'temp_eval_run.py')
                with open(temp_file, 'w') as f:
                    f.write(code)

                # Execute
                try:
                    run_proc = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=30)
                    stdout_str = run_proc.stdout
                    stderr_str = run_proc.stderr

                    if run_proc.returncode != 0:
                        error_msg = f"Execution failed with return code {run_proc.returncode}:\n{stderr_str}"
                        tier = "failed"
                    else:
                        # Exit code 0 -> at least 'runs'
                        tier = "runs"
                        if is_type_safe:
                            tier = "type-safe"

                        # Validate semantic assertions from task declaration
                        # Supports either 'assertions' list/dict or legacy 'validation_script'
                        task_assertions = task.get('assertions')
                        validation_script = task.get('validation_script')

                        if task_assertions:
                            # Evaluate task-declared assertions
                            val_env = {"stdout": stdout_str, "stderr": stderr_str}
                            all_asserts_passed = True
                            for a_expr in task_assertions:
                                try:
                                    if not eval(a_expr, {"__builtins__": __builtins__, **val_env}):
                                        all_asserts_passed = False
                                        error_msg = f"Assertion failed: {a_expr}"
                                        break
                                except Exception as ae:
                                    all_asserts_passed = False
                                    error_msg = f"Assertion evaluation error: {ae}"
                                    break
                            if all_asserts_passed:
                                tier = "semantically-validated"
                                passed = True
                        elif validation_script:
                            # Pass stdout and stderr to the validation env
                            val_env = {"stdout": stdout_str, "stderr": stderr_str}
                            try:
                                exec(validation_script, {"__builtins__": __builtins__, **val_env})
                                tier = "semantically-validated"
                                passed = True
                            except AssertionError as ae:
                                error_msg = f"Validation failed: {ae}"
                            except Exception as e:
                                error_msg = f"Validation script crashed: {e}\n{traceback.format_exc()}"
                        else:
                            # No declared assertions: caps at type-safe
                            passed = (tier == "type-safe")

                        # Determine pass attribution structurally
                        # If passed code is byte-identical post-normalization to emitted code -> path
                        if tier in ("runs", "type-safe", "semantically-validated"):
                            pass_attribution = "path"

                except subprocess.TimeoutExpired:
                    error_msg = "Execution timed out after 30 seconds."
                    tier = "failed"
                except Exception as e:
                    error_msg = f"Failed to run code: {e}"
                    tier = "failed"

            result = {
                "profile": profile,
                "embedder": emb,
                "llm": llm,
                "task_id": task["task_id"],
                "tier": tier,
                "passed": (tier == "semantically-validated") or (passed and tier == "type-safe"),
                "pass_attribution": pass_attribution,
                "error": error_msg,
                "stdout": stdout_str,
                "code": code,
                "repaired_code": repaired_code
            }
            results.append(result)
            print(f"Task {task['task_id']} Tier: {tier}, Passed: {result['passed']}, Attribution: {pass_attribution}")
            if not result['passed']:
                print(f"Error: {error_msg}")
        
        print("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            subprocess.run(["fuser", "-k", f"{API_PORT}/tcp"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(3)

    with open(os.path.join(str(PROJECT_ROOT), 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluations completed. Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_eval()
