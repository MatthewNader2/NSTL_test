import json
import os
import time
import urllib.request
import subprocess
from tqdm import tqdm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import find_llama_server, MODELS_DIR

def get_local_model():
    model_dir = os.path.join(MODELS_DIR, "llms", "Qwen2.5-Coder-7B-Instruct-GGUF")
    model_path = os.path.join(model_dir, "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    
    if not os.path.exists(model_path) and os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if "q4_k_m" in f.lower() and f.endswith(".gguf"):
                model_path = os.path.join(model_dir, f)
                break
                
    return model_path

def run_local_harvester(skeleton_file, output_file, model_path, batch_start=0, batch_size=None):
    server_exe = find_llama_server() or "llama-server"
    
    print(f"[*] Starting Standalone CUDA llama-server on RTX...")
    # BALANCED HARDWARE LIMITS (ANTI-OCP TRIP):
    # -ngl 22: Fast GPU processing but still bottlenecks slightly on CPU transfer.
    # --threads 6: Keeps CPU utilization reasonable.
    # -b 1: Batch size 1. The absolute ONLY way to prevent the prompt-processing wattage spike.
    server_log = open("server_log.txt", "w")
    server_proc = subprocess.Popen(
        [server_exe, "-m", model_path, "-c", "1024", "-ngl", "6", "--threads", "4", "-b", "16", "--port", "8080"],
        stdout=server_log,
        stderr=subprocess.STDOUT
    )
    
    print("[*] Waiting for server to load model into VRAM (polling health endpoint)...")
    server_ready = False
    deadline = time.time() + 60  # 60 second timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            print("[FATAL] Server crashed during load! Check server_log.txt")
            server_log.close()
            return
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            server_ready = True
            break
        except Exception:
            time.sleep(2)
    
    if not server_ready:
        print("[FATAL] Server failed to become ready within 60 seconds! Check server_log.txt")
        server_proc.terminate()
        server_log.close()
        return
    
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
            f"# Auto-synthesized by Qwen 7B Harvester (Standalone CUDA Server) from {skeleton_file}",
            "import typing",
            "import pandas as pd",
            "import numpy as np",
            "import cv2",
            ""
        ]
        # Initialize file to ensure clean incremental saves
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

    print(f"[*] Processing {len(skeletons)} skeletons at ultra-speed via local HTTP API...")
    start_time = time.time()
    
    try:
        for skel in tqdm(skeletons):
            if server_proc.poll() is not None:
                print("\n[FATAL] llama-server.exe crashed unexpectedly! Aborting to save time.")
                break
                
            name = skel["name"]
            parent = skel["contexts"][0]
            
            prompt = prompt_template.format(
                parent=parent,
                name=name,
                params=", ".join(skel["params"])
            )
            
            # OpenAI-compatible API request
            req_data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ast_node",
                        "schema": json_schema
                    }
                }
            }
            
            req = urllib.request.Request(
                "http://127.0.0.1:8080/v1/chat/completions",
                data=json.dumps(req_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    res_body = json.loads(response.read().decode())
                    response_text = res_body['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"[!] API Error on {parent}.{name}: {e}")
                if hasattr(e, 'read'):
                    print(e.read().decode())
                continue
                
            try:
                var = json.loads(response_text)
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
                # Incremental save to prevent data loss on power failure
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(stub + "\n")

            except Exception as e:
                print(f"[!] Error processing {parent}.{name}: {e}")
                pass
                
            # HARDWARE PROTECTION: Sleep for 3 seconds between requests
            # This gives the laptop GPU a "breather" and prevents transient 
            # power spikes from tripping the power supply OCP.
            time.sleep(3)
            
    finally:
        print("[*] Terminating llama-server...")
        server_proc.terminate()
        server_proc.wait()
        server_log.close()

    end_time = time.time()
        
    print(f"[+] Successfully harvested skeletons into {output_file}")
    print(f"    - Total Inference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton", help="Input skeleton JSON file")
    parser.add_argument("output", help="Output python file")
    parser.add_argument("--batch-start", type=int, default=0, help="Starting index")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    model_path = get_local_model()
    run_local_harvester(args.skeleton, args.output, model_path, args.batch_start, args.batch_size)
