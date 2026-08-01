import json
import re
import sys
import os
import time
import subprocess
from tqdm import tqdm

def setup_environment():
    print("[*] Setting up Colab environment...")
    
    print("[*] Checking for GPU...")
    res = subprocess.run(["nvidia-smi"], check=False)
    if res.returncode != 0:
        print("[!] FATAL: No NVIDIA GPU detected! Aborting to prevent CPU fallback.")
        sys.exit(1)
        
    # 1. Install llama-cpp-python with CUDA support
    print("[*] Installing llama-cpp-python (cuBLAS)...")
    subprocess.run(
        "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --upgrade --force-reinstall --no-cache-dir",
        shell=True, check=True
    )
    
    # 2. Install huggingface_hub
    subprocess.run([
        sys.executable, "-m", "pip", "install", "huggingface_hub"
    ], check=True)

def download_model():
    print("[*] Downloading Qwen2.5-Coder-7B-Instruct-GGUF (Q4_K_M)...")
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    )
    return model_path

def run_harvester(skeleton_file, output_file, model_path, batch_start=0, batch_size=None):
    from llama_cpp import Llama
    
    print("[*] Loading 7B Model into Colab T4 VRAM...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1, # Offload all to T4
        verbose=False
    )
    
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
        
    if batch_size is not None:
        skeletons = skeletons[batch_start:batch_start + batch_size]
        
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            start_index = content.count("\ndef ")
            if content.startswith("def "): start_index += 1
            
    if start_index > 0:
        print(f"[*] Resuming from index {start_index} (Skipping already generated nodes)...")
        skeletons = skeletons[start_index:]
    else:
        headers = [
            f"# Auto-synthesized by Qwen 7B Harvester on Google Colab from {skeleton_file}",
            "import typing",
            "import pandas as pd",
            "import numpy as np",
            "import cv2",
            ""
        ]
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(headers) + "\n")

    prompt_template = """You are an AST parameter generator for {parent}.{name}.
Params: {params}

Generate 1 configuration variant JSON object.
Rules for args_string:
- ONLY use literal values (numbers, strings, True/False) or valid library constants (e.g. cv2.COLOR_BGR2GRAY).
- NEVER use imaginary variable names. If a parameter requires a dynamic runtime variable, you MUST leave args_string empty "".

JSON Keys:
- suffix: A short, unique identifier describing the variant. If args_string is empty, use "default".
- args_string: The exact string of python arguments. 
- keywords: 3-5 relevant search keywords.
"""

    json_schema = {
        "type": "object",
        "properties": {
            "suffix": {"type": "string"},
            "args_string": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["suffix", "args_string", "keywords"]
    }

    print(f"[*] Processing {len(skeletons)} skeletons...")
    start_time = time.time()
    
    for i, skel in enumerate(tqdm(skeletons)):
        name = skel["name"]
        parent = skel["contexts"][0]
        
        prompt = prompt_template.format(
            parent=parent,
            name=name,
            params=", ".join(skel["params"])
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            response_format={
                "type": "json_object",
                "schema": json_schema
            }
        )
        
        response_text = response['choices'][0]['message']['content'].strip()
        
        try:
            # Strip markdown fences that LLMs commonly wrap JSON in
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```"):
                # Remove opening fence (```json or ```)
                cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text)
                # Remove closing fence
                cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            var = json.loads(cleaned_text)
            suffix = var.get("suffix", "default")
            args_str = var.get("args_string", "")
            keywords = ", ".join(var.get("keywords", [name]))
            
            safe_name = f"{parent}_{name}_{suffix}".replace(".", "_")
            
            if skel["is_method"]:
                if args_str:
                    invocation = f"output_var = input_var.{name}({args_str})"
                else:
                    invocation = f"output_var = input_var.{name}()"
            else:
                if args_str:
                    invocation = f"output_var = {parent}.{name}(input_var, {args_str})"
                else:
                    invocation = f"output_var = {parent}.{name}(input_var)"
                    
            stub = f'''def {safe_name}(input_var: 'any') -> 'any_computed':
    """
    Keywords: {parent}.{name}, {keywords}
    """
    {invocation}
'''
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(stub + "\n")
                
            # Write progress update to file
            progress = {
                "total": len(skeletons) + start_index,
                "completed": i + 1 + start_index,
                "current_node": name
            }
            with open("progress.json", "w", encoding="utf-8") as f:
                json.dump(progress, f)
                
        except Exception as e:
            print(f"Failed to parse node {name}: {e}")

    end_time = time.time()
        
    print(f"[+] Successfully harvested {len(skeletons)} skeletons into {output_file}")
    print(f"    - Total Inference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    skeleton = "skeleton_cv2.json"
    output = "qwen_cv2.py"
    
    setup_environment()
    model_path = download_model()
    run_harvester(skeleton, output, model_path)
