import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import time
import main
from main import initialize_engine, run_prompt, InitRequest, RunRequest
from router import HardwareProfiler

def run_main_test(profile: str):
    main.TREES_DIR = "trees_test"
    print(f"\n==================================================")
    print(f" TESTING main.py IMPLEMENTATION WITH PROFILE {profile}")
    print(f"==================================================")
    
    t0 = time.time()
    try:
        init_res = initialize_engine(InitRequest(profile=profile, device="auto"))
        print(f"  [+] Engine initialized in {time.time() - t0:.2f}s: {init_res}")
        
        prompt = "Write a simple Python function to add two numbers."
        print(f"  [>] Prompt: {prompt}")
        
        t1 = time.time()
        run_res = run_prompt(RunRequest(prompt=prompt))
        print(f"  [+] Run completed in {time.time() - t1:.2f}s")
        print(f"  [+] Generated Code length: {len(run_res.get('code', ''))}")
        print(f"  [+] First 200 chars of code:\n{run_res.get('code', '')[:200]}")
        print("  [SUCCESS]")
    except Exception as e:
        import traceback
        print(f"  [FAILED] Crashed!")
        traceback.print_exc()

if __name__ == "__main__":
    run_main_test("A")
    run_main_test("B")
    run_main_test("C")
