import requests
import time
import json
import subprocess

def run_test():
    print("Starting main.py server...")
    server_process = subprocess.Popen(["python", "main.py"])
    
    status_url = "http://127.0.0.1:58102/api/status"
    run_url = "http://127.0.0.1:58102/api/run"
    
    print("Waiting for server to be ready...")
    ready = False
    start_time = time.time()
    while time.time() - start_time < 300: # Wait up to 5 minutes
        try:
            resp = requests.get(status_url)
            if resp.status_code == 200 and resp.json().get("status") == "ready":
                ready = True
                break
        except Exception:
            pass
        time.sleep(5)
        
    if not ready:
        print("Server failed to become ready.")
        server_process.kill()
        return
        
    print("Server is ready. Sending prompt...")
    prompt = "make a code that loads dataset in csv in a sample name you provide (to test if it fetches parameters correctly), and to remove nulls, normalize values, perform ML preprocessing for the data ( a vague request in the middle for testing), then perfrom PCA and finally training an sklearn model with this data (to see if it can and if it would make train-test-valid split and other stuff)"
    
    payload = {"prompt": prompt}
    
    try:
        response = requests.post(run_url, json=payload, timeout=600)
        data = response.json()
        print("========== GENERATED CODE ==========")
        print(data.get("code", "No code returned."))
        print("========== END CODE ==========")
        print("\nLogs:")
        for log in data.get("logs", []):
            print(f"[{log.get('type')}] {log.get('msg')}")
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        server_process.kill()

if __name__ == "__main__":
    run_test()
