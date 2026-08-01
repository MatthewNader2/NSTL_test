import os
import sys
import json
import urllib.request
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import find_llama_server, MODELS_DIR, HARVESTS_DIR

server_exe = find_llama_server()
model_path = os.path.join(MODELS_DIR, "llms", "Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
skel_file = os.path.join(HARVESTS_DIR, "skeleton_cv2.json")

if not server_exe or not os.path.exists(server_exe):
    print(f"[SKIP] llama-server executable not found ({server_exe})")
    sys.exit(0)

if not os.path.exists(model_path):
    print(f"[SKIP] GGUF model not found ({model_path})")
    sys.exit(0)

if not os.path.exists(skel_file):
    print(f"[SKIP] Skeleton file not found ({skel_file})")
    sys.exit(0)

server_proc = subprocess.Popen(
    [server_exe, "-m", model_path, "-c", "1024", "-ngl", "22", "--port", "8080"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(10)

with open(skel_file, 'r', encoding='utf-8') as f:
    skeletons = json.load(f)[:5]

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

for skel in skeletons:
    name = skel["name"]
    prompt = f"Generate JSON for {name}"
    
    req_data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
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
            print(f"[{name}] API SUCCESS")
            print(res_body['choices'][0]['message']['content'])
    except Exception as e:
        print(f"[{name}] API ERROR: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode())

server_proc.terminate()
