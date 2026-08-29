"""
Tier 2 Docstring Enricher (No LLM, Zero Hallucination)

Enriches Tier 1 structural cells in trees/*.json with real prose descriptions
extracted from introspected Python libraries (pandas, sklearn, cv2, matplotlib,
numpy, scipy, python_core) using docstring_parser and inspect.getdoc().

Sets:
  - cell["docstring"] = clean_summary
  - cell["enrichment_source"] = "docs"
  - cell["enriched_at"] = ISO8601 timestamp

Does NOT modify any parameter types, ports, code_template, stage, or dependencies.
"""

from __future__ import annotations
import importlib
import inspect
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docstring_parser

# Suppress deprecation warnings from old dynamic imports during introspection
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_cv2_doxygen_doc(doc_str: str) -> str:
    """Parse OpenCV C++/Doxygen docstring for brief summary."""
    if not doc_str:
        return ""

    lines = doc_str.split("\n")
    summary_lines = []

    for line in lines:
        cleaned = line.strip().lstrip(".").strip()
        if cleaned.startswith("@brief"):
            summary_lines.append(cleaned.replace("@brief", "").strip())
        elif not summary_lines and cleaned and not cleaned.startswith("cvtColor") and not cleaned.startswith("@") and "(" not in cleaned:
            summary_lines.append(cleaned)

    summary = " ".join(summary_lines).strip()
    if not summary and lines:
        for l in lines:
            cl = l.strip().lstrip(".").strip()
            if cl and "(" not in cl and not cl.startswith("@"):
                summary = cl
                break

    return summary


def clean_docstring_summary(raw_doc: str, domain: str) -> str:
    """Extracts a clean, single-paragraph or single-sentence summary from raw docstring."""
    if not raw_doc or not raw_doc.strip():
        return ""

    if domain == "cv2":
        summary = parse_cv2_doxygen_doc(raw_doc)
        if summary:
            return summary

    try:
        parsed = docstring_parser.parse(raw_doc)
        summary = parsed.short_description or ""
        if parsed.long_description and len(summary) < 25:
            # Append first sentence of long description if short description is minimal
            first_sentence = parsed.long_description.split("\n\n")[0].replace("\n", " ").strip()
            if first_sentence and first_sentence != summary:
                summary = (summary + " " + first_sentence).strip() if summary else first_sentence
    except Exception:
        # Fallback to first non-empty line
        lines = [l.strip() for l in raw_doc.strip().splitlines() if l.strip()]
        summary = lines[0] if lines else ""

    # Clean formatting
    summary = re.sub(r"\s+", " ", summary).strip()
    # Strip common Sphinx/NumPyDoc formatting artifacts e.g. ````, `:class:`, `:func:`
    summary = re.sub(r":(class|func|meth|mod|attr|ref):`([^`]+)`", r"\2", summary)
    summary = re.sub(r"`([^`]+)`", r"\1", summary)
    return summary


def resolve_symbol_from_cell(cell: Dict[str, Any], default_domain: str) -> Tuple[Optional[Any], str]:
    """Resolves the live Python object and docstring corresponding to a Cell dict."""
    # Method 1: Check dependencies (e.g. from sklearn.preprocessing import StandardScaler)
    deps = cell.get("dependencies", [])
    for dep in deps:
        m = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import\s+([a-zA-Z0-9_]+)", dep)
        if m:
            mod_name, obj_name = m.groups()
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, obj_name):
                    obj = getattr(mod, obj_name)
                    doc = inspect.getdoc(obj) or getattr(obj, "__doc__", "") or ""
                    if doc:
                        return obj, doc
            except Exception:
                pass

    # Method 2: Inspect code_template (e.g. pd.read_csv, plt.plot, cv2.cvtColor)
    tmpl = cell.get("code_template", "")
    for m in re.finditer(r"([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)\(", tmpl):
        full_call = m.group(1)
        parts = full_call.split(".")
        root_map = {
            "pd": "pandas",
            "np": "numpy",
            "plt": "matplotlib.pyplot",
            "cv2": "cv2",
            "sklearn": "sklearn",
            "scipy": "scipy"
        }
        actual_root = root_map.get(parts[0], parts[0])
        try:
            curr = importlib.import_module(actual_root)
            for p in parts[1:]:
                curr = getattr(curr, p)
            doc = inspect.getdoc(curr) or getattr(curr, "__doc__", "") or ""
            if doc:
                return curr, doc
        except Exception:
            pass

    # Method 3: Cell ID path traversal through module hierarchy
    cid = cell.get("cell_id", "")
    clean_id = re.sub(r"_(DEFAULT|INPUT|OUTPUT)$", "", cid)
    parts = clean_id.split("_")

    domains_to_try = [default_domain]
    if default_domain == "matplotlib":
        domains_to_try = ["matplotlib.pyplot", "matplotlib"]
    elif default_domain == "python_core":
        domains_to_try = ["builtins", "math", "os", "sys"]

    for domain_mod in domains_to_try:
        try:
            curr = importlib.import_module(domain_mod)
        except Exception:
            continue

        sub_parts = parts[1:] if parts[0].lower() == default_domain.lower() else parts
        
        # Try full attribute name match
        for i in range(len(sub_parts)):
            target = "_".join(sub_parts[i:]).lower()
            for attr in dir(curr):
                if attr.lower() == target or attr.lower().replace("_", "") == target.replace("_", ""):
                    obj = getattr(curr, attr)
                    doc = inspect.getdoc(obj) or getattr(obj, "__doc__", "") or ""
                    if doc:
                        return obj, doc

        # Try step-by-step navigation
        for i in range(len(sub_parts)):
            step_obj = curr
            for sp in sub_parts[i:]:
                found = None
                for attr in dir(step_obj):
                    if attr.lower() == sp.lower() or attr.lower().replace("_", "") == sp.lower().replace("_", ""):
                        found = getattr(step_obj, attr)
                        break
                if found is not None:
                    step_obj = found
                else:
                    step_obj = None
                    break
            if step_obj is not None and step_obj is not curr:
                doc = inspect.getdoc(step_obj) or getattr(step_obj, "__doc__", "") or ""
                if doc:
                    return step_obj, doc

    return None, ""


def enrich_tree_file(tree_path: Path) -> Tuple[int, int, int]:
    """
    Enriches a single trees/{domain}.json file.
    Returns: (total_cells, previously_enriched, newly_enriched)
    """
    domain = tree_path.stem
    print(f"[*] Processing domain '{domain}' from {tree_path}...")

    with open(tree_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    cells = tree_data.get("cells", [])
    total_cells = len(cells)
    previously_enriched = 0
    newly_enriched = 0

    now_iso = datetime.now(timezone.utc).isoformat()

    for cell in cells:
        existing_doc = (cell.get("docstring") or "").strip()
        existing_source = cell.get("enrichment_source")

        # Keep existing curated docstrings
        if existing_doc and cell.get("source_priority", 100) <= 10:
            previously_enriched += 1
            continue

        if existing_doc and existing_source:
            previously_enriched += 1
            continue

        # Try resolving real docstring
        obj, raw_doc = resolve_symbol_from_cell(cell, domain)
        if raw_doc:
            summary = clean_docstring_summary(raw_doc, domain)
            if summary and len(summary) > 5:
                cell["docstring"] = summary
                cell["enrichment_source"] = "docs"
                cell["enriched_at"] = now_iso
                newly_enriched += 1
            elif existing_doc:
                previously_enriched += 1
        elif existing_doc:
            previously_enriched += 1

    # Atomic write back to disk
    tmp_path = tree_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, tree_path)

    print(f"  [+] Domain '{domain}': {total_cells} total | {previously_enriched} prior | {newly_enriched} newly enriched from docs")
    return total_cells, previously_enriched, newly_enriched


def enrich_all_trees(trees_dir: Path) -> None:
    json_files = sorted(trees_dir.glob("*.json"))
    print(f"[*] Running Tier 2 Docstring Enrichment across {len(json_files)} trees in {trees_dir}...")

    grand_total = 0
    grand_prior = 0
    grand_new = 0

    for jf in json_files:
        try:
            total, prior, newly = enrich_tree_file(jf)
            grand_total += total
            grand_prior += prior
            grand_new += newly
        except Exception as e:
            print(f"[!] Error processing {jf.name}: {e}")

    print("\n" + "=" * 70)
    print(f"[*] Tier 2 Docs Enrichment Complete:")
    print(f"    Total Cells:            {grand_total:,}")
    print(f"    Prior Docstrings:       {grand_prior:,}")
    print(f"    Newly Enriched (Docs):  {grand_new:,}")
    print(f"    Total Coverage:         {(grand_prior + grand_new) / max(grand_total, 1) * 100:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    target_dir = PROJECT_ROOT / "trees"
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.is_file():
            enrich_tree_file(arg_path)
            sys.exit(0)
        elif arg_path.is_dir():
            target_dir = arg_path

    enrich_all_trees(target_dir)
