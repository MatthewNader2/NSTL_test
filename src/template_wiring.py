"""
src/template_wiring.py

Core shared functions for maintaining and repairing the NSTL template wiring invariant:
1. Every {placeholder} in `code_template` must have a matching key in `inputs`
   (except `output_var`).
2. Every declared input in `inputs` must be referenced in `code_template`.

Zero regular expressions.
"""

from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Set


def clean_malformed_template_braces(template: str) -> str:
    """Cleans malformed double braces or trailing artifacts in templates without regex."""
    if not template or "{" not in template:
        return template
    cleaned = template
    while "{{" in cleaned or "}}" in cleaned:
        cleaned = cleaned.replace("{{", "{").replace("}}", "}")
    for artifact in (", {/*}", "{/*},", "{/*}", ", {\\*}", "{\\*},", "{\\*}"):
        cleaned = cleaned.replace(artifact, "")
    return cleaned


def extract_placeholders(template: str) -> List[str]:
    """Extracts all {var} placeholder names from a template string without regex."""
    placeholders: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == '{':
            j = template.find('}', i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier():
                    placeholders.append(inner)
                i = j + 1
                continue
        i += 1
    return placeholders


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

# Name shapes that strongly indicate a scalar / container / callable type.
# These are STRATIFIED HEURISTICS over identifier morphology, not domain
# vocabulary. They are intentionally small and conservative; unknown names
# resolve to "any".
_INT_NAMES: Set[str] = {
    "n", "k", "count", "size", "length", "len", "num", "number",
    "index", "idx", "position", "pos", "iteration", "iter",
    "epoch", "epochs", "batch", "batch_size", "seed",
    "top_k", "max_iter", "n_iter", "num_iter", "depth", "width",
    "degree", "rank", "order", "level", "step", "steps", "stride",
}

_FLOAT_NAMES: Set[str] = {
    "rate", "alpha", "beta", "gamma", "lambda_", "threshold", "thresh",
    "eps", "epsilon", "lr", "learning_rate", "ratio", "weight",
    "score", "prob", "probability", "p", "momentum", "decay",
    "temperature", "scale", "factor", "tolerance", "tol", "epsilon_",
}

_BOOL_NAMES: Set[str] = {
    "flag", "verbose", "enabled", "enable", "debug", "shuffle",
    "dropna", "drop_na", "normalize", "normalized", "fit_intercept",
    "copy", "inplace", "in_place", "use_gpu", "cuda", "parallel",
    "strict", "optional", "required", "sorted", "reverse", "ascending",
    "descending", "closed", "unique", "drop", "keep",
}

_STR_NAMES: Set[str] = {
    "path", "filepath", "file_path", "filename", "file_name",
    "dir", "directory", "folder", "name", "label", "text",
    "message", "msg", "pattern", "column", "col", "key", "tag",
    "title", "description", "desc", "url", "uri", "format",
    "encoding", "mode", "kind", "type", "strategy", "method",
    "separator", "sep", "delimiter", "delim", "prefix", "suffix",
    "regex", "template", "symbol",
}

_DATAFRAME_NAMES: Set[str] = {
    "data", "df", "dataset", "frame", "table", "rows", "records",
    "corpus", "documents",
}

_ARRAY_NAMES: Set[str] = {
    "arr", "array", "values", "val", "vec", "vector", "tensor",
    "x", "y", "z", "xs", "ys", "samples", "features", "labels",
    "series", "matrix", "mat", "embedding", "embeddings",
}

_MODEL_NAMES: Set[str] = {
    "model", "estimator", "clf", "classifier", "regressor",
    "net", "network", "pipeline", "transformer", "encoder", "decoder",
}

_CALLABLE_NAMES: Set[str] = {
    "func", "fn", "callback", "callable", "predicate", "transform",
    "mapper", "reducer", "comparator", "key_func", "key_fn",
}

_PREFIXES: tuple = ("input_", "in_", "arg_", "param_")
_SUFFIXES: tuple = ("_input", "_in", "_arg", "_param", "_value", "_val")


def _canonicalize_param_name(param_name: str) -> str:
    name = str(param_name).strip().lower()
    for pre in _PREFIXES:
        if name.startswith(pre) and len(name) > len(pre):
            name = name[len(pre):]
            break
    for suf in _SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            name = name[: -len(suf)]
            break
    return name


def infer_port_type(param_name: str, domain: str = "generic") -> str:
    """
    Infers the canonical type_name for a given parameter name using
    identifier-shape heuristics. Unknown names resolve to "any".
    """
    if not param_name:
        return "any"
    name = _canonicalize_param_name(param_name)
    if not name:
        return "any"

    if name in _INT_NAMES:
        return "int"
    if name in _FLOAT_NAMES:
        return "float"
    if name in _BOOL_NAMES:
        return "bool"
    if name in _STR_NAMES:
        return "str"
    if name in _DATAFRAME_NAMES:
        return "DataFrame"
    if name in _ARRAY_NAMES:
        return "array"
    if name in _MODEL_NAMES:
        return "model"
    if name in _CALLABLE_NAMES:
        return "callable"

    # Morphological hints: prefixes that strongly indicate a type family.
    if name.endswith("_path") or name.endswith("_file") or name.endswith("_name"):
        return "str"
    if name.endswith("_count") or name.endswith("_size") or name.endswith("_num"):
        return "int"
    if name.endswith("_flag") or name.startswith("is_") or name.startswith("has_") or name.startswith("use_"):
        return "bool"
    if name.startswith("num_") or name.startswith("n_"):
        return "int"
    if name.endswith("_list") or name.endswith("_arr") or name.endswith("_array"):
        return "array"
    if name.endswith("_df") or name.startswith("df_"):
        return "DataFrame"

    return "any"


def transform_call_ast_with_flag(code_snippet: str, mod_alias: str, flag_attr: str) -> str:
    """Uses AST Node Transformation to inject a flag attribute into a function call snippet dynamically."""
    if not code_snippet:
        return f"{{output_var}} = {mod_alias}.{flag_attr}()"

    cleaned_snippet = clean_malformed_template_braces(code_snippet)

    placeholders = list(dict.fromkeys(extract_placeholders(cleaned_snippet)))
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
                    ctx=ast.Load(),
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

        for safe_id, ph in ph_map.items():
            unparsed = unparsed.replace(safe_id, f"{{{ph}}}")

        return clean_malformed_template_braces(unparsed)
    except Exception:
        return cleaned_snippet


def wire_default_flag(cell: Any, code: str, mod_alias: str) -> str:
    """Injects default flag attributes via AST without text surgery hacks."""
    slots = getattr(cell, "slots", {}) if not isinstance(cell, dict) else cell.get("slots", {})
    tree_meta = getattr(cell, "tree_meta", {}) if not isinstance(cell, dict) else cell.get("tree_meta", {})
    flag_attr = (slots or {}).get("default_flag_attr") or (tree_meta or {}).get("default_flag_attr")
    if not flag_attr:
        return code
    return transform_call_ast_with_flag(code, mod_alias, str(flag_attr))


def _port_is_preservable(spec: Any) -> bool:
    """
    A port is preservable (i.e. must NOT be pruned even if unreferenced by the
    template) when it is declared optional, config-role, or variadic. These
    ports are supplied out-of-band (via **kwargs or configuration dicts) and
    deleting them would destroy the port contract.
    """
    if not isinstance(spec, dict):
        return False
    if not spec.get("required", True):
        return True
    if spec.get("kind") == "config" or spec.get("role") == "config":
        return True
    if spec.get("variadic") in ("kwargs", "kw", "**kwargs"):
        return True
    if spec.get("supplied_out_of_band") is True:
        return True
    return False


def repair_wiring_invariant(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Enforces the template wiring invariant on a cell without regex."""
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
                    "kind": item.get("kind"),
                    "role": item.get("role"),
                    "variadic": item.get("variadic"),
                    "supplied_out_of_band": item.get("supplied_out_of_band", False),
                }
        cell["inputs"] = inputs_dict
        modified = True
    elif not isinstance(raw_inputs, dict):
        cell["inputs"] = {}
        modified = True

    inputs = cell["inputs"]
    placeholders = set(extract_placeholders(cell.get("code_template", ""))) - {"output_var"}

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

    # Prune declared inputs that are not referenced in the template, EXCEPT
    # optional / config / variadic ports which may be supplied out-of-band.
    for inp in list(inputs.keys()):
        if inp in placeholders:
            continue
        if _port_is_preservable(inputs[inp]):
            continue
        del inputs[inp]
        modified = True

    return modified
