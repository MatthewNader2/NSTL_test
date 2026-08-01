import os
print("Launching background daemon...")
os.system("nohup python cloud_harvester.py > nohup.out 2>&1 &")
print("Done!")
