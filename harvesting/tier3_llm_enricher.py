"""
Tier 3 Narrow, Schema-Constrained LLM Metadata Pass

Given Tier 1+2 grounded node objects, this module enriches nodes ONLY with
semantic search keywords and refined prose descriptions.

Constraints:
- Output MUST be JSON schema-constrained: {"cell_id": "...", "keywords": [...], "description": "..."}
- Batch 20-50 nodes per prompt/call.
- Never writes types, params, or code templates.
- Merges keywords/description into the base structural/enriched nodes. Discards any unapproved fields.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIER3_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": {"type": "string"}
                },
                "required": ["cell_id", "keywords", "description"]
            }
        }
    },
    "required": ["items"]
}


def generate_fallback_keywords(node: dict) -> list[str]:
    """Extract baseline semantic keywords directly from function name and docstring."""
    name = node.get("name", "")
    cell_id = node.get("cell_id", "")
    domain = node.get("domain", "")
    desc = node.get("description", "")

    keywords = set()
    keywords.add(name)
    keywords.add(f"{domain}.{name}")

    if "COLOR" in cell_id or "cvt" in name.lower():
        keywords.update(["image", "color", "conversion", "format"])
    if "blur" in name.lower() or "filter" in name.lower():
        keywords.update(["image", "filter", "smoothing", "blur"])
    if "edge" in name.lower() or "canny" in name.lower():
        keywords.update(["image", "edge", "detection", "contours"])
    if "threshold" in name.lower():
        keywords.update(["image", "binary", "threshold", "segmentation"])
    if "read" in name.lower() or "load" in name.lower():
        keywords.update(["io", "read", "load", "file"])
    if "write" in name.lower() or "save" in name.lower():
        keywords.update(["io", "write", "save", "export"])

    words = re.findall(r"[a-zA-Z]{3,}", desc)
    for w in words[:5]:
        keywords.add(w.lower())

    return sorted(list(keywords))


def process_batch_llm(batch: list[dict], llama_url: str = "http://127.0.0.1:8080/v1/chat/completions") -> list[dict]:
    """Process a batch of 20-50 nodes with schema-constrained decoding."""
    batch_summaries = []
    for node in batch:
        batch_summaries.append({
            "cell_id": node["cell_id"],
            "name": node.get("name"),
            "params": [p["name"] for p in node.get("params", [])],
            "description": node.get("description", "")
        })

    prompt = f"""You are an expert search keyword and summary generator for code APIs.
Generate high-quality search keywords (3-6 per item) and a concise summary description for each cell_id.

Input items:
{json.dumps(batch_summaries, indent=2)}

Output JSON schema strictly matching:
{{"items": [{{"cell_id": "...", "keywords": ["..."], "description": "..."}}]}}
"""

    req_data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "tier3_metadata_batch",
                "schema": TIER3_JSON_SCHEMA
            }
        }
    }

    try:
        req = urllib.request.Request(
            llama_url,
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = json.loads(response.read().decode())
            content = res_body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("items", [])
    except Exception:
        # Fallback to local heuristic metadata if server is offline or times out
        results = []
        for node in batch:
            results.append({
                "cell_id": node["cell_id"],
                "keywords": generate_fallback_keywords(node),
                "description": node.get("description", "")
            })
        return results


def merge_tier3_metadata(base_nodes: list[dict], llm_metadata_items: list[dict]) -> list[dict]:
    """Merge Tier 3 metadata into base nodes while strictly discarding any unapproved fields."""
    meta_map = {item["cell_id"]: item for item in llm_metadata_items if isinstance(item, dict) and "cell_id" in item}

    for node in base_nodes:
        cell_id = node["cell_id"]
        meta = meta_map.get(cell_id, {})

        # Safety Check: Discard unapproved authority fields if model hallucinated them
        for forbidden in ("type", "params", "inputs", "outputs", "code", "domain_implementations"):
            if forbidden in meta:
                print(f"[!] Warning: Discarded model hallucinated field '{forbidden}' for {cell_id}")

        keywords = meta.get("keywords")
        if keywords and isinstance(keywords, list):
            node["keywords"] = keywords
        elif "keywords" not in node:
            node["keywords"] = generate_fallback_keywords(node)

        if meta.get("description") and len(meta["description"]) > len(node.get("description", "")):
            node["description"] = meta["description"]

        node["provenance"] = "tier1_2_3_enriched"

    return base_nodes


def run_tier3_enrichment(input_enriched_json: str, output_final_json: str, batch_size: int = 25, limit: int = None):
    print(f"[*] Running Tier 3 Metadata Enrichment on {input_enriched_json}...")
    with open(input_enriched_json, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if limit:
        nodes = nodes[:limit]

    all_llm_items = []
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        llm_items = process_batch_llm(batch)
        all_llm_items.extend(llm_items)

    final_nodes = merge_tier3_metadata(nodes, all_llm_items)

    os.makedirs(os.path.dirname(os.path.abspath(output_final_json)), exist_ok=True)
    with open(output_final_json, "w", encoding="utf-8") as f:
        json.dump(final_nodes, f, indent=2)

    print(f"[+] Saved {len(final_nodes)} Tier 3 enriched nodes to {output_final_json}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tier3_llm_enricher.py <input_enriched.json> <output_final.json> [batch_size] [limit]")
        sys.exit(1)

    batch_sz = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else None
    run_tier3_enrichment(sys.argv[1], sys.argv[2], batch_sz, lim)
