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

def clean_malformed_template_braces(template: str) -> str:
    """Cleans malformed double braces, regex patterns, or trailing commas in templates."""
    if not template or "{" not in template:
        return template
    cleaned = re.sub(r"\{\{(\w+)\}(\w+)\}", r"{\1\2}", template)
    cleaned = re.sub(r"\{(\w+)\{(\w+)\}\}", r"{\1\2}", cleaned)
    cleaned = re.sub(r"\{\{(\w+)\}\}", r"{\1}", cleaned)
    cleaned = re.sub(r",\s*\{[/\\*]\}", "", cleaned)
    cleaned = re.sub(r"\{[/\\*]\},\s*", "", cleaned)
    cleaned = re.sub(r"\{[/\\*]\}", "", cleaned)
    cleaned = re.sub(r"\{([a-zA-Z_]\w*)=[^}]*\}", r"{\1}", cleaned)
    cleaned = re.sub(r"\{(\w+)\)\}", r"{\1}", cleaned)
    while re.search(r"\{\{\w+\}\w*\}", cleaned) or re.search(r"\{\w*\{\w+\}\}", cleaned):
        cleaned = re.sub(r"\{\{(\w+)\}(\w+)\}", r"{\1\2}", cleaned)
        cleaned = re.sub(r"\{(\w+)\{(\w+)\}\}", r"{\1\2}", cleaned)
        cleaned = re.sub(r"\{\{(\w+)\}\}", r"{\1}", cleaned)
    return cleaned


def infer_port_type(param_name: str, domain: str = "generic") -> str:
    """Infers the canonical type_name for a given parameter name."""
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
        if f"{mod_alias}." in cleaned_snippet and "_DEFAULT" in cleaned_snippet:
            return re.sub(rf"{mod_alias}\.\w+_DEFAULT", f"{mod_alias}.{flag_attr}", cleaned_snippet)
        return cleaned_snippet


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

    # Ensure 1-to-1 input key alignment with template placeholder if single port differs
    if len(inputs) == 1 and len(placeholders) == 1:
        inp_key = next(iter(inputs.keys()))
        ph_key = next(iter(placeholders))
        if inp_key != ph_key:
            inputs[ph_key] = inputs.pop(inp_key)
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
