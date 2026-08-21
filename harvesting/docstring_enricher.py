"""
Tier 2 Docstring Enricher (No LLM, No Hallucination)

Enriches Tier 1 structural node objects with prose descriptions and per-parameter
documentation extracted from docstrings using docstring_parser / numpydoc or
C++/Doxygen docstring parser for OpenCV.

Does NOT modify any parameter types, required flags, or code structure.
"""

import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
import docstring_parser

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_cv2_doxygen_doc(doc_str: str) -> tuple[str, dict[str, str]]:
    """Parse OpenCV C++/Doxygen docstring for brief summary and @param descriptions."""
    if not doc_str:
        return "", {}

    summary = ""
    param_docs = {}

    lines = doc_str.split("\n")
    summary_lines = []

    for line in lines:
        cleaned = line.strip().lstrip(".").strip()
        if cleaned.startswith("@brief"):
            summary_lines.append(cleaned.replace("@brief", "").strip())
        elif cleaned.startswith("@param"):
            match = re.match(r"@param\s+([a-zA-Z0-9_]+)\s+(.*)", cleaned)
            if match:
                p_name, p_desc = match.groups()
                param_docs[p_name] = p_desc.strip()
        elif not summary_lines and cleaned and not cleaned.startswith("cvtColor") and not cleaned.startswith("@") and "(" not in cleaned:
            summary_lines.append(cleaned)

    summary = " ".join(summary_lines).strip()
    if not summary and lines:
        # Fallback to first non-signature line
        for l in lines:
            cl = l.strip().lstrip(".").strip()
            if cl and "(" not in cl and not cl.startswith("@"):
                summary = cl
                break

    return summary, param_docs


def enrich_cv2_nodes(nodes: list[dict]) -> list[dict]:
    """Enrich OpenCV structural nodes using live cv2 docstrings."""
    import cv2

    for node in nodes:
        func_name = node.get("name")
        if not func_name:
            cell_id = node.get("cell_id", "")
            func_name = cell_id.replace("CV2_", "").replace("_DEFAULT", "").lower()

        obj = getattr(cv2, func_name, None)
        doc = getattr(obj, "__doc__", "") if obj else ""

        summary, param_docs = parse_cv2_doxygen_doc(doc)
        node["description"] = summary or f"OpenCV {func_name} routine."

        for param in node.get("params", []):
            p_name = param["name"]
            if p_name in param_docs:
                param["param_doc"] = param_docs[p_name]
            else:
                param["param_doc"] = f"Argument {p_name} for cv2.{func_name}."

    return nodes


def enrich_python_library_nodes(nodes: list[dict], library_name: str) -> list[dict]:
    """Enrich pure Python library structural nodes using docstring_parser."""
    try:
        mod = importlib.import_module(library_name)
    except ImportError:
        print(f"[!] Warning: Could not import {library_name} for docstring enrichment.")
        return nodes

    for node in nodes:
        cell_id = node.get("cell_id", "")
        func_name = node.get("name")

        obj = None
        # Attempt to resolve function object
        if hasattr(mod, func_name):
            obj = getattr(mod, func_name)
        elif "." in cell_id:
            parts = cell_id.split("_")
            curr = mod
            for p in parts:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                elif hasattr(curr, p.lower()):
                    curr = getattr(curr, p.lower())
            if curr is not mod:
                obj = curr

        doc = inspect.getdoc(obj) if obj else ""
        if not doc and obj and hasattr(obj, "__doc__"):
            doc = getattr(obj, "__doc__", "") or ""

        parsed = docstring_parser.parse(doc)
        summary = parsed.short_description or ""
        if parsed.long_description and len(summary) < 20:
            summary = (summary + " " + parsed.long_description).strip()

        node["description"] = summary or f"{library_name} {func_name} function."

        param_doc_map = {p.arg_name: p.description for p in parsed.params if p.arg_name and p.description}

        for param in node.get("params", []):
            p_name = param["name"]
            if p_name in param_doc_map:
                param["param_doc"] = param_doc_map[p_name].strip()
            else:
                param["param_doc"] = f"Parameter {p_name} for {func_name}."

    return nodes


def enrich_nodes(input_file: str, output_file: str, library_name: str):
    print(f"[*] Running Tier 2 Docstring Enricher for '{library_name}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if library_name == "cv2":
        enriched = enrich_cv2_nodes(nodes)
    else:
        enriched = enrich_python_library_nodes(nodes, library_name)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"[+] Enriched {len(enriched)} nodes saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python docstring_enricher.py <input_structural.json> <output_enriched.json> <library_name>")
        sys.exit(1)

    enrich_nodes(sys.argv[1], sys.argv[2], sys.argv[3])
