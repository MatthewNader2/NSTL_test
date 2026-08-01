import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
target_dir = os.path.join(PROJECT_ROOT, "models", "llms", "Qwen2.5-Coder-7B-Instruct-GGUF")
os.makedirs(target_dir, exist_ok=True)

api = HfApi()
files = api.list_repo_files(repo_id=repo_id)

# Find all files related to q4_k_m
q4_files = [f for f in files if "q4_k_m" in f.lower() and f.endswith(".gguf")]

if not q4_files:
    print("Could not find q4_k_m model in the repo.")
else:
    for file in q4_files:
        print(f"Downloading {file} to {target_dir}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
    print("Download complete!")
