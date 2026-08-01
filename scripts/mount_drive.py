import subprocess
import time
import sys

print("[*] Starting drivemount process...")
p = subprocess.Popen(["colab", "drivemount", "-s", "final_harvester"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait 10 seconds to allow the process to initialize and connect to Colab
print("[*] Waiting 10 seconds before sending Enter key...")
time.sleep(10)

print("[*] Sending Enter key to process...")
p.stdin.write("\n")
p.stdin.flush()

print("[*] Reading output from process...")
while True:
    line = p.stdout.readline()
    if not line and p.poll() is not None:
        break
    if line:
        print(line.strip())

print(f"[*] Process exited with code {p.returncode}")
