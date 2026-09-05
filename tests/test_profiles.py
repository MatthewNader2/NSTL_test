import sys
import os
import time
import urllib.request
import json
import threading
import subprocess

EMBEDDINGS = ["jina-embeddings-v5-text-nano", "embeddinggemma-300m"]
LLMS = ["qwen2.5-coder-0.5b-instruct", "qwen2.5-coder-1.5b-instruct", "Qwen2.5-Coder-7B-Instruct-GGUF"]

MATRIX = []
# Profile A uses embedding only
MATRIX.append(("A", "auto"))
# Profile C and D on all LLMs
for llm in LLMS:
    MATRIX.append(("C", llm))
    MATRIX.append(("D", llm))

PROMPT = "Read a CSV file named data.csv into a pandas dataframe, drop any rows with missing values, sort it by the 'age' column in descending order, and then save the cleaned dataframe to a new CSV file named cleaned_data.csv."

REPORT_FILE = "test_report.md"

def write_report(content):
    with open(REPORT_FILE, "a") as f:
        f.write(content + "\n")

if os.path.exists(REPORT_FILE):
    os.remove(REPORT_FILE)

write_report("# NSTL Prototype Benchmarking Report")
write_report(f"**Prompt**: `{PROMPT}`\n")

for emb_model in EMBEDDINGS:
    write_report(f"## Embedding Model: {emb_model}")
    print(f"\n######################################################################")
    print(f"STARTING EMBEDDING MODEL: {emb_model}")
    print(f"######################################################################")
    
    for profile, llm_model in MATRIX:
        print(f"\n======================================")
        print(f"TESTING PROFILE {profile} (LLM: {llm_model}, EMB: {emb_model})")
        print(f"======================================")
        
        write_report(f"### Profile {profile} | LLM: {llm_model}")
        
        # Start server with TEST_HEADLESS=1 to bypass pywebview UI
        env = os.environ.copy()
        env["TEST_HEADLESS"] = "1"
        proc = subprocess.Popen(["python3", "src/main.py"], env=env, cwd="/media/matthew/New Volume/grad_test/nstl_prototype")
        
        # Wait for server to respond on 58102
        print("Waiting for server to boot...")
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:58102/api/status", timeout=1)
                break
            except Exception:
                time.sleep(1)
                
        print("Server up! Sending initialize...")
        # Init
        init_payload = {"profile": profile, "embedder_model": emb_model, "llm_model": llm_model}
        req = urllib.request.Request("http://127.0.0.1:58102/api/initialize", data=json.dumps(init_payload).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
        except Exception as e:
            print(f"Initialization failed: {e}")
            proc.terminate()
            proc.wait()
            write_report(f"**Error**: Initialization failed - {e}\n")
            continue
        
        # Wait for ready
        ready = False
        start_wait = time.time()
        while time.time() - start_wait < 1200:
            try:
                resp = json.loads(urllib.request.urlopen("http://127.0.0.1:58102/api/status").read().decode())
                if resp.get("status") == "ready":
                    ready = True
                    break
            except:
                pass
            time.sleep(1)
            
        if not ready:
            print("Server failed to become ready in time.")
            proc.terminate()
            proc.wait()
            write_report(f"**Error**: Server failed to become ready (timeout).\n")
            continue
            
        print(f"Profile {profile} ready. Running prompt...")
        
        run_payload = {"prompt": PROMPT}
        req = urllib.request.Request("http://127.0.0.1:58102/api/run", data=json.dumps(run_payload).encode(), headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
            code = resp.get("code", "No code generated!")
            print(f"\n[Profile {profile}] Output Code:\n{code}")
            
            write_report(f"```python\n{code}\n```\n")
        except Exception as e:
            print(f"Error: {e}")
            write_report(f"**Error**: Generation failed - {e}\n")
            
        print("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        time.sleep(2) # brief pause between runs
