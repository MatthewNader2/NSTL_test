import sys, os, time, urllib.request, json, threading, subprocess

profile = "D"
llm_model = "qwen2.5-coder-1.5b-instruct"

print(f"\n======================================")
print(f"TESTING PROFILE {profile} (LLM: {llm_model})")
print(f"======================================")

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

print("Server up! Sending initialize...")
init_payload = {"profile": profile, "embedder_model": "auto", "llm_model": llm_model}
req = urllib.request.Request("http://127.0.0.1:58102/api/initialize", data=json.dumps(init_payload).encode(), headers={"Content-Type": "application/json"})
urllib.request.urlopen(req)

while True:
    try:
        resp = json.loads(urllib.request.urlopen("http://127.0.0.1:58102/api/status").read().decode())
        if resp.get("status") == "ready":
            break
    except:
        pass
    time.sleep(1)

print(f"Profile {profile} ready. Running prompt...")
prompt = "Read a CSV file named data.csv into a pandas dataframe, drop any rows with missing values, sort it by the 'age' column in descending order, and print the first 5 rows."
run_payload = {"prompt": prompt}
req = urllib.request.Request("http://127.0.0.1:58102/api/run", data=json.dumps(run_payload).encode(), headers={"Content-Type": "application/json"})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    print(f"\n[Profile {profile}] Output Code:\n")
    print(resp.get("code", "No code generated!"))
except Exception as e:
    print(f"Error: {e}")

proc.terminate()
proc.wait()
