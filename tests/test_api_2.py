import os
import sys
import json
import urllib.request
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import find_llama_server, MODELS_DIR

server_exe = find_llama_server()
model_path = os.path.join(MODELS_DIR, "llms", "Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")

if not server_exe or not os.path.exists(server_exe):
    print(f"[SKIP] llama-server executable not found ({server_exe})")
    sys.exit(0)

if not os.path.exists(model_path):
    print(f"[SKIP] GGUF model not found ({model_path})")
    sys.exit(0)

server_proc = subprocess.Popen(
    [server_exe, "-m", model_path, "-c", "1024", "-ngl", "22", "--port", "8080"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(10)

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

req_data = {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10,
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
        print("Success:", response.read().decode())
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'read'):
        print(e.read().decode())
finally:
    server_proc.terminate()
