import os
if os.path.exists("nohup.out"):
    with open("nohup.out", "r") as f:
        print(f.read())
else:
    print("nohup.out not found yet")
    
if os.path.exists("progress.json"):
    with open("progress.json", "r") as f:
        print("PROGRESS:", f.read())
