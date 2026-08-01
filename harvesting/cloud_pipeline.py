import os
import sys

# Make sure we can import local modules
sys.path.append('/content')

import extract_skeleton
from cloud_harvester import run_harvester

LIBRARIES = ['builtins', 'cv2', 'pandas', 'numpy', 'sklearn', 'requests', 'networkx']
DRIVE_DIR = "/content/drive/MyDrive/NSTL_Harvester"

def main():
    if not os.path.exists(DRIVE_DIR):
        print(f"[*] Creating directory: {DRIVE_DIR}")
        os.makedirs(DRIVE_DIR, exist_ok=True)
        
    model_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct-GGUF/snapshots/10f7a79e60ea9b491a9d18e5473fcbc9810f6071/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    
    # Pre-download the model to save time during loops if missing
    if not os.path.exists(model_path):
        print("[*] Pre-downloading 7B model...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        )
        
    for lib in LIBRARIES:
        print(f"\n==============================================")
        print(f"[*] Processing Library: {lib}")
        print(f"==============================================")
        
        skeleton_file = os.path.join(DRIVE_DIR, f"skeleton_{lib}.json")
        output_file = os.path.join(DRIVE_DIR, f"qwen_{lib}.py")
        
        # 1. Extract Skeleton
        if not os.path.exists(skeleton_file):
            print(f"[*] Extracting skeleton for {lib}...")
            try:
                extract_skeleton.extract_skeleton(lib, skeleton_file)
            except Exception as e:
                print(f"[!] Failed to extract skeleton for {lib}: {e}")
                continue
        else:
            print(f"[*] Skeleton {skeleton_file} already exists, skipping extraction.")
            
        # 2. Harvest with LLM
        if not os.path.exists(output_file):
            print(f"[*] Starting LLM Harvester for {lib}...")
            try:
                run_harvester(skeleton_file, output_file, model_path, batch_start=0, batch_size=None)
                print(f"[+] Successfully harvested {lib}!")
            except Exception as e:
                print(f"[!] Harvester failed for {lib}: {e}")
        else:
            print(f"[*] Output {output_file} already exists, skipping harvest.")

if __name__ == "__main__":
    main()
