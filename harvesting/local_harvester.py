"""
harvesting/local_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Generates validated keyword parameter configurations for structural skeletons using local LLM inference.
"""

from __future__ import annotations
import json
import os
import time
import urllib.request
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import MODELS_DIR


def run_local_harvester(skeleton_file: str, output_file: str, batch_start: int = 0, batch_size: int = None):
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)

    if batch_size is not None:
        skeletons = skeletons[batch_start:batch_start + batch_size]

    prompt_template = """You are an AST parameter generator for {parent}.{name}.
Parameters: {params}

Generate 1 valid keyword configuration object.
Rules for args_string:
- ONLY use literal values (numbers, strings, True/False) or standard library constants (e.g. cv2.INTER_LINEAR).
- If no keyword defaults are needed, leave args_string empty "".

JSON Schema:
{{
  "suffix": "short_unique_descriptor_or_default",
  "args_string": "param1=value, param2=value",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

    json_schema = {
        "type": "object",
        "properties": {
            "suffix": {"type": "string"},
            "args_string": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["suffix", "args_string", "keywords"]
    }

    results = []
    print(f"[*] Processing {len(skeletons)} skeletons via local inference server...")

    for skel in tqdm(skeletons):
        name = skel["name"]
        parent = skel["contexts"][0]
        params_str = ", ".join(p["name"] if isinstance(p, dict) else str(p) for p in skel.get("params", []))

        prompt = prompt_template.format(parent=parent, name=name, params=params_str)
        req_data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
            "response_format": {"type": "json_object", "schema": json_schema}
        }

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8080/v1/chat/completions",
                data=json.dumps(req_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10.0) as response:
                res_body = json.loads(response.read().decode())
                response_text = res_body['choices'][0]['message']['content'].strip()
                var_dict = json.loads(response_text)
        except Exception:
            var_dict = {"suffix": "default", "args_string": "", "keywords": [name]}

        suffix = var_dict.get("suffix", "default")
        args_str = var_dict.get("args_string", "")
        keywords = var_dict.get("keywords", [name])

        domain = parent.split(".")[0]
        safe_id = f"{parent}_{name}_{suffix}".upper().replace(".", "_")

        # Code template construction
        if skel["is_method"]:
            call_code = f"{{input_var}}.{name}({args_str})" if args_str else f"{{input_var}}.{name}()"
        else:
            call_code = f"{parent}.{name}({{input_var}}, {args_str})" if args_str else f"{parent}.{name}({{input_var}})"

        # Infer real domain types (no more 'any' soup)
        default_t = "DataFrame" if "pandas" in domain else ("Mat" if "cv2" in domain else "ndarray")

        results.append({
            "cell_id": safe_id,
            "domain_name": domain,
            "node_type": "function",
            "node_role": "function",
            "stage": 1 if "read" in name or "load" in name else (3 if "to_" in name or "write" in name or "save" in name else 2),
            "keywords": keywords + [name.lower(), domain],
            "inputs": {"input_data": {"type_name": default_t, "state": "raw"}},
            "outputs": {"output_data": {"type_name": default_t, "state": "computed"}},
            "dependencies": [f"import {domain}"],
            "code_template": f"{{output_var}} = {call_code}"
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Successfully harvested {len(results)} variant nodes into {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton", help="Input skeleton JSON")
    parser.add_argument("output", help="Output JSON file")
    args = parser.parse_args()
    run_local_harvester(args.skeleton, args.output)
