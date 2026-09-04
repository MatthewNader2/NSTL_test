"""
src/template_wiring.py

Core shared functions for maintaining and repairing the NSTL template wiring invariant:
1. Every {placeholder} in `code_template` must have a matching key in `inputs`
   (except `output_var`).
2. Every declared input in `inputs` must be referenced in `code_template`.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Set

DOMAIN_CONTAINERS: Dict[str, str] = {
    "pandas": "DataFrame",
    "numpy": "ndarray",
    "scipy": "ndarray",
    "sklearn": "ndarray",
    "cv2": "Mat",
    "matplotlib": "Figure",
    "python_core": "any",
}

# Extensible registry: maps (domain, param_semantic_role) → canonical type_name.
# To support a new domain, add rows here — no control-flow changes needed.
DOMAIN_PARAM_TYPES: Dict[tuple, str] = {
    ("pandas", "index"):     "Index",
    ("pandas", "columns"):   "Index",
    ("sklearn", "model"):    "BaseEstimator",
    ("sklearn", "estimator"): "BaseEstimator",
}

PARAM_TYPE_HINTS: Dict[str, str] = {
    "src": "Mat",
    "img": "Mat",
    "image": "Mat",
    "probImage": "Mat",
    "window": "Rect",
    "rect": "Rect",
    "criteria": "TermCriteria",
    "ksize": "tuple",
    "threshold": "float",
    "threshold1": "float",
    "threshold2": "float",
    "epsilon": "float",
    "dest_path": "str",
    "filepath": "str",
    "path": "str",
    "filename": "str",
}


def clean_malformed_template_braces(template: str) -> str:
    """Fixes nested braces caused by naive substring replacement, e.g.:

    {{image}Points} -> {imagePoints}
    {{src}1} -> {src1}
    {{var}} -> {var}
    """
    if not template or "{" not in template:
        return template

    # Handle {{name}suffix} -> {namesuffix}
    cleaned = re.sub(r"\{\{(\w+)\}(\w+)\}", r"{\1\2}", template)
    # Handle {prefix{name}} -> {prefixname}
    cleaned = re.sub(r"\{(\w+)\{(\w+)\}\}", r"{\1\2}", cleaned)
    # Handle {{name}} -> {name}
    cleaned = re.sub(r"\{\{(\w+)\}\}", r"{\1}", cleaned)

    # Strip non-identifier braces like {/} and {*}
    cleaned = re.sub(r",\s*\{[/\\*]\}", "", cleaned)
    cleaned = re.sub(r"\{[/\\*]\},\s*", "", cleaned)
    cleaned = re.sub(r"\{[/\\*]\}", "", cleaned)

    # Normalize default values inside braces: {param=default} -> {param}
    cleaned = re.sub(r"\{([a-zA-Z_]\w*)=[^}]*\}", r"{\1}", cleaned)

    # Fix unmatched parentheses inside braces like {2)} -> {2}
    cleaned = re.sub(r"\{(\w+)\)\}", r"{\1}", cleaned)

    # Repeat if deeper nesting exists
    while re.search(r"\{\{\w+\}\w*\}", cleaned) or re.search(r"\{\w*\{\w+\}\}", cleaned):
        cleaned = re.sub(r"\{\{(\w+)\}(\w+)\}", r"{\1\2}", cleaned)
        cleaned = re.sub(r"\{(\w+)\{(\w+)\}\}", r"{\1\2}", cleaned)
        cleaned = re.sub(r"\{\{(\w+)\}\}", r"{\1}", cleaned)

    return cleaned


def infer_port_type(param_name: str, domain: str = "generic") -> str:
    """Infers the canonical type_name for a given parameter name and domain.
    Uses declarative registry lookups — no domain-specific if/elif chains."""
    if param_name in PARAM_TYPE_HINTS:
        return PARAM_TYPE_HINTS[param_name]

    lowered = param_name.lower()
    if lowered in ("data", "arr", "array", "x", "y", "b", "u", "v", "input_var"):
        return DOMAIN_CONTAINERS.get(domain, "ndarray")

    # Declarative registry lookup: (domain, semantic_role) → type
    registry_type = DOMAIN_PARAM_TYPES.get((domain, lowered))
    if registry_type:
        return registry_type

    if lowered.startswith("is_") or lowered.startswith("has_"):
        return "bool"

    return "any"


def transform_call_ast_with_flag(code_snippet: str, mod_alias: str, flag_attr: str) -> str:
    """Uses AST Node Transformation to inject a flag attribute into a function call snippet dynamically.

    Avoids global string replacements that corrupt identifiers like imagePoints -> {{image}Points}.
    """
    if not code_snippet:
        return f"{{output_var}} = {mod_alias}.{flag_attr}()"

    cleaned_snippet = clean_malformed_template_braces(code_snippet)

    # Map placeholders to valid Python identifiers for AST parsing
    placeholders = list(set(re.findall(r"\{(\w+)\}", cleaned_snippet)))
    safe_code = cleaned_snippet
    ph_map = {}
    for i, ph in enumerate(placeholders):
        safe_id = f"__ph_{i}_{ph}__"
        ph_map[safe_id] = ph
        safe_code = safe_code.replace(f"{{{ph}}}", safe_id)

    try:
        tree = ast.parse(safe_code)

        class FlagASTReplacer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                flag_node = ast.Attribute(
                    value=ast.Name(id=mod_alias, ctx=ast.Load()),
                    attr=flag_attr,
                    ctx=ast.Load()
                )
                if node.args:
                    node.args[-1] = flag_node
                elif node.keywords:
                    node.keywords[-1].value = flag_node
                else:
                    node.args.append(flag_node)
                return node

        transformed = FlagASTReplacer().visit(tree)
        ast.fix_missing_locations(transformed)
        unparsed = ast.unparse(transformed)

        # Restore original placeholders exactly
        for safe_id, ph in ph_map.items():
            unparsed = unparsed.replace(safe_id, f"{{{ph}}}")

        return clean_malformed_template_braces(unparsed)
    except Exception:
        # Fallback: simple token replacement if AST fails
        if "cv2." in cleaned_snippet and "_DEFAULT" in cleaned_snippet:
            return re.sub(r"cv2\.\w+_DEFAULT", f"{mod_alias}.{flag_attr}", cleaned_snippet)
        return cleaned_snippet


def repair_cv2_variant_cell(
    cell: Dict[str, Any],
    cell_map: Dict[str, Dict[str, Any]],
    base_default_map: Dict[str, Dict[str, Any]],
) -> bool:
    """Repairs cv2 enum-variant cells with argument-less fake calls (Bug B).

    Returns True if the cell was modified.
    """
    cid = cell.get("cell_id", "")
    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    cell["code_template"] = tmpl

    m = re.match(r"\{output_var\}\s*=\s*cv2\.([A-Za-z0-9_]+)\(\)", tmpl)
    if not m:
        return False

    parts = cid.rsplit("_", 1)
    flag = parts[1] if len(parts) > 1 else ""
    grp_def_id = parts[0] + "_DEFAULT"

    target_template = None
    source_cell = None

    # Strategy 1: Check group default sibling (e.g. CV2_CAMSHIFT_ALGO_HINT_DEFAULT)
    if grp_def_id in cell_map and not cell_map[grp_def_id].get("code_template", "").endswith("()"):
        grp_tmpl = clean_malformed_template_braces(cell_map[grp_def_id].get("code_template", ""))
        m_flag = re.search(r"cv2\.([A-Za-z0-9_]+_DEFAULT)", grp_tmpl)
        if m_flag:
            def_flag = m_flag.group(1)
            prefix = def_flag.rsplit("_", 1)[0]
            variant_flag = f"{prefix}_{flag}"
            target_template = grp_tmpl.replace(f"cv2.{def_flag}", f"cv2.{variant_flag}")
            source_cell = cell_map[grp_def_id]
        else:
            target_template = grp_tmpl
            source_cell = cell_map[grp_def_id]

    # Strategy 2: Check base function default (e.g. CV2_HOUGHLINESPOINTSET_DEFAULT)
    if not target_template:
        found_base = None
        for b in sorted(base_default_map.keys(), key=len, reverse=True):
            if cid.startswith(b + "_") or cid == b:
                found_base = b
                break
        if found_base:
            base_cell = base_default_map[found_base]
            base_tmpl = clean_malformed_template_braces(base_cell.get("code_template", ""))
            flag_attr = cid[len(found_base) + 1 :] if cid != found_base else ""
            if flag_attr:
                target_template = transform_call_ast_with_flag(base_tmpl, "cv2", flag_attr)
            else:
                target_template = base_tmpl
            source_cell = base_cell

    # Strategy 3: _GROUP_ or _GROUP reflection recovery
    if not target_template:
        if "_GROUP_" in cid:
            fn_name = cid.split("_GROUP_")[-1]
            target_template = f"{{output_var}} = cv2.{fn_name}({{input_var}})"
            source_cell = cell
        elif cid.endswith("_GROUP"):
            fn_name = cid.replace("CV2_", "").replace("_GROUP", "")
            target_template = f"{{output_var}} = cv2.{fn_name}({{input_var}})"
            source_cell = cell

    if not target_template:
        return False

    cell["code_template"] = clean_malformed_template_braces(target_template)

    # Inherit inputs and outputs from the source sibling
    if source_cell and source_cell is not cell:
        cell["inputs"] = dict(source_cell.get("inputs", {}))
        cell["outputs"] = dict(source_cell.get("outputs", {}))

    return True


def repair_wiring_invariant(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Enforces the template wiring invariant on a cell:

    1. Fixes Bug A: renames input 'X' to 'input_var' when '{input_var}' is in template.
    2. Ensures every placeholder in `code_template` (except output_var) is declared in `inputs`.
    3. Prunes unused declared inputs not referenced in `code_template`.

    Returns True if the cell was modified.
    """
    modified = False

    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    if tmpl != cell.get("code_template", ""):
        cell["code_template"] = tmpl
        modified = True

    # Normalize inputs to dict format if needed
    raw_inputs = cell.get("inputs")
    if isinstance(raw_inputs, list):
        inputs_dict = {}
        for item in raw_inputs:
            if isinstance(item, dict):
                pname = item.get("name", "input_var")
                inputs_dict[pname] = {
                    "type_name": item.get("type", item.get("type_name", "any")),
                    "state": item.get("state", "any"),
                    "required": item.get("required", True),
                    "default_value": item.get("default_value", None),
                    "description": item.get("description", ""),
                }
        cell["inputs"] = inputs_dict
        modified = True
    elif not isinstance(raw_inputs, dict):
        cell["inputs"] = {}
        modified = True

    inputs = cell["inputs"]
    placeholders = set(re.findall(r"\{(\w+)\}", cell.get("code_template", ""))) - {"output_var"}

    # Bug A: Single source of truth for input keys vs placeholders
    if "X" in inputs and "input_var" in placeholders and "input_var" not in inputs:
        inputs["input_var"] = inputs.pop("X")
        modified = True

    # Ensure every placeholder has a matching declared input port
    for ph in placeholders:
        if ph not in inputs:
            port_type = infer_port_type(ph, domain)
            inputs[ph] = {
                "type_name": port_type,
                "state": "any",
                "required": True,
                "default_value": None,
                "description": f"Input {ph}",
            }
            modified = True

    # Prune declared inputs that are not referenced in the template
    for inp in list(inputs.keys()):
        if inp not in placeholders:
            del inputs[inp]
            modified = True

    return modified
