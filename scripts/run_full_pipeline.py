import os
import subprocess
import time
import sys
import json

LIBRARIES = ['cv2', 'pandas', 'numpy', 'sklearn', 'requests', 'networkx']
SESSION_NAME = "gpu_harvester_v2"
BATCH_SIZE = 50
POLL_TIMEOUT_MINUTES = 60

def run_colab_automation():
    print(f"==================================================")
    print(f"[*] Starting Batched Colab Orchestration")
    print(f"==================================================")
    
    os.makedirs("harvests", exist_ok=True)
    
    # 1. Start Session
    print(f"\n[1/4] Provisioning T4 GPU Colab Instance...")
    subprocess.run(["colab", "new", "-s", SESSION_NAME, "--gpu", "T4"], check=True)
    
    try:
        # 2. Upload Files
        print(f"\n[2/4] Uploading core scripts to Colab...")
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, "tools/extract_skeleton.py", "/content/extract_skeleton.py"], check=True)
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, "cloud_harvester.py", "/content/cloud_harvester.py"], check=True)
        
        # 3. Process each library
        print(f"\n[3/4] Beginning library harvest loop...")
        for lib in LIBRARIES:
            out_skeleton = f"skeleton_{lib}.json"
            local_skeleton = f"harvests/{out_skeleton}"
            
            print(f"\n--------------------------------------------------")
            print(f"    Processing: {lib}")
            print(f"--------------------------------------------------")
            
            # Step A: Extract Skeleton if not already downloaded
            if not (os.path.exists(local_skeleton) and os.path.getsize(local_skeleton) > 0):
                print(f"[*] Extracting {lib} skeleton on VM...")
                extract_trigger = f"""import os
os.chdir('/content')
os.system('python extract_skeleton.py {lib} {out_skeleton}')
"""
                with open("trigger_extract.py", "w") as f:
                    f.write(extract_trigger)
                subprocess.run(["colab", "exec", "-s", SESSION_NAME, "-f", "trigger_extract.py"])
                
                print(f"[*] Downloading {out_skeleton}...")
                subprocess.run(["colab", "download", "-s", SESSION_NAME, f"/content/{out_skeleton}", local_skeleton], check=True)
            
            # Step B: Read Skeleton to determine batches
            with open(local_skeleton, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            total_nodes = len(skeletons)
            print(f"[*] Total AST nodes for {lib}: {total_nodes}")
            
            # Step C: Loop through batches
            for batch_start in range(0, total_nodes, BATCH_SIZE):
                out_py = f"qwen_{lib}_{batch_start}.py"
                local_py = f"harvests/{out_py}"
                
                if os.path.exists(local_py) and os.path.getsize(local_py) > 0:
                    print(f"    [+] Skipping batch {batch_start} to {batch_start+BATCH_SIZE}: Already downloaded!")
                    continue
                
                print(f"    [*] Processing batch {batch_start} to {min(batch_start+BATCH_SIZE, total_nodes)}...")
                
                trigger_code = f"""import os
os.chdir('/content')
print('Starting Batch {batch_start}')
os.system('python cloud_harvester.py {out_skeleton} {out_py} --batch-start {batch_start} --batch-size {BATCH_SIZE}')
"""
                with open("trigger_batch.py", "w") as f:
                    f.write(trigger_code)
                    
                print(f"        -> Executing batch on VM...")
                # Start execution (we don't check=True as it might drop websocket connection while running)
                subprocess.run(["colab", "exec", "-s", SESSION_NAME, "-f", "trigger_batch.py"])
                
                print(f"        -> Polling for {out_py}...")
                max_attempts = (POLL_TIMEOUT_MINUTES * 60) // 30
                attempts = 0
                success = False
                while attempts < max_attempts:
                    time.sleep(30)
                    attempts += 1
                    sys.stdout.write(f"\r           ... Polling ({attempts * 30}s elapsed) ... ")
                    sys.stdout.flush()
                    
                    # Try to download
                    result = subprocess.run(
                        ["colab", "download", "-s", SESSION_NAME, f"/content/{out_py}", local_py], 
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        print(f"\n        [+] SUCCESS! Downloaded {local_py}")
                        success = True
                        break
                    
                    # Fail fast if session disconnected
                    stderr_lower = result.stderr.lower() if result.stderr else ""
                    stdout_lower = result.stdout.lower() if result.stdout else ""
                    
                    # Ensure it's a SESSION not found, not just the file missing
                    if ("session" in stdout_lower and "not found" in stdout_lower) or ("session" in stderr_lower and "not found" in stderr_lower):
                        print(f"\n        [!] FATAL: Colab session '{SESSION_NAME}' disconnected or crashed!")
                        return # Abort the entire pipeline so user can restart
                
                if not success:
                    print(f"\n        [!] Timeout waiting for batch {batch_start}. Aborting pipeline.")
                    return
                    
    finally:
        # 4. Clean up
        print(f"\n[4/4] Pipeline concluded.")
        for tmp in ["trigger_extract.py", "trigger_batch.py"]:
            if os.path.exists(tmp):
                os.remove(tmp)
            
    print(f"\n[+] Full Pipeline run completed! All batches are saved in the 'harvests/' folder.")

if __name__ == "__main__":
    run_colab_automation()
