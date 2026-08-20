# unification.py
import os
import re
import sys
import ast
import importlib.util
from typing import Optional, Set, Dict, Tuple, Any, List
from dataclasses import dataclass, field
from log_config import get_logger


logger = get_logger('unification')

# We need AlgebraicSignature imported. It's safe to import it dynamically or statically if lattice.py is in same dir.
from lattice import AlgebraicSignature


CANONICAL_IMPORT_MAP = {
    "pd": ("import pandas as pd\nimport pandas", "pandas"),
    "pandas": ("import pandas\nimport pandas as pd", "pandas"),
    "np": ("import numpy as np\nimport numpy", "numpy"),
    "numpy": ("import numpy\nimport numpy as np", "numpy"),
    "cv2": ("import cv2", "cv2"),
    "plt": ("import matplotlib.pyplot as plt", "matplotlib"),
    "matplotlib": ("import matplotlib.pyplot as plt", "matplotlib"),
    "sns": ("import seaborn as sns", "seaborn"),
    "seaborn": ("import seaborn as sns", "seaborn"),
    "sklearn": ("import sklearn", "sklearn"),
    "sk": ("import sklearn", "sklearn"),
    "scipy": ("import scipy", "scipy"),
    "sp": ("import scipy", "scipy"),
    "faiss": ("import faiss", "faiss"),
    "torch": ("import torch", "torch"),
    "nx": ("import networkx as nx", "networkx"),
}


KNOWN_PLACEHOLDERS: Set[str] = set()

_UNIFY_PLACEHOLDERS = frozenset({
    "input_var", "output_var", "input_filename", "output_filename", "input_source"
})



class UnresolvedPlaceholderError(Exception):
    pass


def assert_placeholders_resolved(template: str, bindings: dict = None, known: set = None):
    """
    Pre-flight placeholder-resolution gate.
    Scans template for {...} placeholders and asserts every one is present in bindings or known placeholders.
    Raises UnresolvedPlaceholderError immediately if any placeholder is unbound.
    """
    if not template:
        return
    bindings = bindings or {}
    known = known if known is not None else _UNIFY_PLACEHOLDERS
    found = set(re.findall(r"\{(\w+)\}", template))
    unresolved = found - set(bindings.keys()) - known
    if unresolved:
        raise UnresolvedPlaceholderError(
            f"Unbound placeholder(s) {unresolved} in template: {template!r}"
        )


SLOT_ROLE_MAP = {}


@dataclass
class ExtractedSlots:
    source_uris: List[str] = field(default_factory=list)
    dest_uris: List[str] = field(default_factory=list)
    named_identifiers: List[str] = field(default_factory=list)  # Column names, feature names
    numeric_constants: Dict[str, Any] = field(default_factory=dict)
    operational_flags: Dict[str, bool] = field(default_factory=dict)
    unstructured_literals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_files": self.source_uris,
            "output_files": self.dest_uris,
            "columns": self.named_identifiers,
            "numeric_kwargs": self.numeric_constants,
            "raw_literals": self.unstructured_literals,
            "descending": self.operational_flags.get("descending", False),
            "is_grayscale": self.operational_flags.get("is_grayscale", False),
            "is_hsv": self.operational_flags.get("is_hsv", False),
            "is_rgb": self.operational_flags.get("is_rgb", False),
        }


def resolve_node_slots(template: str, extracted_params: Any = None, context=None, target_cell=None) -> Dict[str, str]:
    """
    Deterministically binds extracted prompt parameters to template slot placeholders.
    Purged of all benchmark fallback strings.
    """

    if not template:
        return {}
    bindings = {}

    if isinstance(extracted_params, ExtractedSlots):
        slots = extracted_params
    elif isinstance(extracted_params, dict):
        slots = ExtractedSlots(
            source_uris=extracted_params.get("input_files", []),
            dest_uris=extracted_params.get("output_files", []),
            named_identifiers=extracted_params.get("columns", []),
            numeric_constants=extracted_params.get("numeric_kwargs", {}),
            operational_flags={
                "descending": extracted_params.get("descending", False),
                "is_grayscale": extracted_params.get("is_grayscale", False),
                "is_hsv": extracted_params.get("is_hsv", False),
                "is_rgb": extracted_params.get("is_rgb", False),
            },
            unstructured_literals=extracted_params.get("raw_literals", [])
        )
    else:
        slots = ExtractedSlots()

    found_placeholders = set(re.findall(r"\{(\w+)\}", template)) - {"input_var", "output_var"}

    for p in found_placeholders:
        role = None
        if target_cell and hasattr(target_cell, 'parameters'):
            for param in target_cell.parameters:
                if param.name == p:
                    role = param.role
                    break

        if not role:
            if p in ("filename", "input_filename", "image_path", "input_path", "source"):
                role = "SOURCE_URI"
            elif p in ("output_filename", "output_path", "dest", "destination"):
                role = "DEST_URI"
            elif p in ("by_column", "by", "column", "col"):
                role = "COLUMN_NAME"
            elif p in ("ascending", "descending"):
                role = "SORT_ORDER"
            elif p in ("code", "color_code", "conversion_code"):
                role = "COLOR_CONV"

        # 1. Source URI resolution
        if role == "SOURCE_URI":
            if slots.source_uris:
                bindings[p] = repr(slots.source_uris[0])
            elif context and hasattr(context, "extracted_parameters") and context.extracted_parameters.get("input_filename"):
                bindings[p] = repr(context.extracted_parameters["input_filename"])
            else:
                # Use find_compatible_variable or fallback
                # Since context doesn't expose type easy here, we just use empty string or lookup
                bindings[p] = "''"

        # 2. Destination URI resolution
        elif role == "DEST_URI":
            if slots.dest_uris:
                bindings[p] = repr(slots.dest_uris[0])
            elif context and hasattr(context, "extracted_parameters") and context.extracted_parameters.get("output_filename"):
                bindings[p] = repr(context.extracted_parameters["output_filename"])
            else:
                bindings[p] = "''"

        # 3. Column references
        elif role == "COLUMN_NAME":
            if slots.named_identifiers:
                bindings[p] = repr(slots.named_identifiers[0])
            else:
                # No column specified — use first column dynamically
                bindings[p] = "{input_var}.columns[0]" if "{input_var}" in template else "0"

        # 4. Sort order boolean
        elif role == "SORT_ORDER":
            descending = slots.operational_flags.get("descending", False)
            bindings[p] = "False" if descending else "True"

        # 5. Color conversion codes
        elif role == "COLOR_CONV":
            if slots.operational_flags.get("is_hsv"):
                bindings[p] = "cv2.COLOR_BGR2HSV"
            elif slots.operational_flags.get("is_rgb"):
                bindings[p] = "cv2.COLOR_BGR2RGB"
            else:
                bindings[p] = "cv2.COLOR_BGR2GRAY"
        else:
            bindings[p] = "None"

    return bindings




def is_module_available(module_name: str) -> bool:
    """Checks if a module exists in stdlib or the active python environment."""
    if not module_name or not module_name.isidentifier():
        return False
    if module_name in sys.stdlib_module_names:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError, AttributeError):
        return False


def resolve_unbound_module(name: str) -> Optional[str]:
    """Dynamically resolves unbound variable/module names to python import statements."""
    if not name or not name.isidentifier():
        return None
    if name in sys.stdlib_module_names or is_module_available(name):
        return f"import {name}"
    alias_map = {
        "pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot",
        "sns": "seaborn", "cv2": "cv2", "sk": "sklearn", "sp": "scipy"
    }
    if name in alias_map:
        base_pkg = alias_map[name].split(".")[0]
        if is_module_available(base_pkg):
            return f"import {alias_map[name]} as {name}"
    return None



class ExecutionContext:
    """Manages variables and literal extraction arguments at runtime using type-state keys."""

    def __init__(self):
        # Format: { variable_name: AlgebraicSignature }
        self.registry: dict[str, AlgebraicSignature] = {}
        self.extracted_parameters = {}
        self.declared_dependencies: Set[str] = set()


    def extract_prompt_parameters(self, user_prompt: str):
        self.extracted_parameters = {}
        # 1. Extract all filenames with extensions
        file_matches = re.findall(
            r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html|txt|jpg|jpeg|png|bmp|tiff|webp|pdf))\b",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if file_matches:
            self.extracted_parameters["input_filename"] = file_matches[0]
            if len(file_matches) > 1:
                self.extracted_parameters["output_filename"] = file_matches[-1]
            self.extracted_parameters["explicit_filename"] = file_matches[0]
        else:
            quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
            if quoted_items:
                self.extracted_parameters["input_filename"] = quoted_items[0]
                if len(quoted_items) > 1:
                    self.extracted_parameters["output_filename"] = quoted_items[-1]
                self.extracted_parameters["explicit_filename"] = quoted_items[0]

        if re.search(r"\b(descending|desc|reverse|highest\s+to\s+lowest)\b", user_prompt.lower()):
            self.extracted_parameters["descending"] = True
        elif re.search(r"\b(ascending|asc|lowest\s+to\s+highest)\b", user_prompt.lower()):
            self.extracted_parameters["descending"] = False

        # 3. Heuristics for arguments
        by_match = re.search(r"\bby\s+(?:the\s+|a\s+|an\s+)*([a-zA-Z_][a-zA-Z0-9_]*)\b", user_prompt, flags=re.IGNORECASE)
        if by_match:
            col_name = by_match.group(1).lower()
            if col_name not in ["descending", "ascending", "the", "a", "an", "column", "columns"]:
                self.extracted_parameters["sort_by"] = col_name
                self.extracted_parameters["by"] = repr(col_name)

        heuristics = []
        all_quoted = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        all_files = set(self.extracted_parameters.values())
        for q in all_quoted:
            if q not in all_files:
                heuristics.append(f"{repr(q)}")
        
        self.extracted_parameters["heuristics"] = heuristics

    def declare_variable(self, name: str, signature: AlgebraicSignature) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        sanitized_name = base_name

        counter = 2
        while sanitized_name in self.registry:
            sanitized_name = f"{base_name}_v{counter}"
            counter += 1

        self.registry[sanitized_name] = signature
        return sanitized_name

    def find_compatible_variable(self, expected_signature: AlgebraicSignature) -> Optional[str]:
        # Priority: Return the most recently declared variable of matching type_name in scope for linear pipeline binding
        for var_name, current_signature in reversed(list(self.registry.items())):
            if current_signature.matches(expected_signature):
                return var_name
        return None


class TopTypeSentinel:
    """Formal Top Type ⊤ sentinel in NSTL type system."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self):
        return "⊤"
    def __str__(self):
        return "any"

TOP_TYPE_SENTINEL = TopTypeSentinel()
TOP_TYPE_SET = frozenset({"any", "Any", "*", "top", "TOP", "ANY", "unknown", "object", TOP_TYPE_SENTINEL})

def is_top_type(t) -> bool:
    if t is TOP_TYPE_SENTINEL or isinstance(t, TopTypeSentinel):
        return True
    if not t:
        return True
    return str(t).strip() in TOP_TYPE_SET

def types_unify(expected_type, actual_type) -> bool:
    """
    Formal unification operator unify(tau_expected, tau_actual).
    Returns True iff tau_expected and tau_actual unify.
    Enforces asymmetric subtyping:
      1. unify(⊤, tau) = True for all tau (Top type wildcard)
      2. unify(tau, ⊤) = True for all tau
      3. unify(tau, tau) = True (Exact identity)
      4. Sub-type containment: Series ⊆ ndarray ⊆ any, Mat ⊆ ndarray ⊆ any, DataFrame ⊆ any
    """
    if is_top_type(expected_type) or is_top_type(actual_type):
        return True
    exp_clean = str(expected_type).strip()
    act_clean = str(actual_type).strip()
    if exp_clean == act_clean:
        return True
    if exp_clean and act_clean and (exp_clean in act_clean or act_clean in exp_clean):
        return True
    exp_l, act_l = exp_clean.lower(), act_clean.lower()
    
    # Asymmetric Subtyping Rules
    subtypes = {
        "series": {"ndarray", "numpy.ndarray", "any", "object"},
        "mat": {"ndarray", "numpy.ndarray", "cv2.mat", "image", "any", "object"},
        "dataframe": {"any", "object"},
        "pd.dataframe": {"any", "object"},
        "pd.series": {"ndarray", "numpy.ndarray", "any", "object"},
    }
    if act_l in subtypes and exp_l in subtypes[act_l]:
        return True
    return False

class UnificationGate:
    """Performs dynamic monadic structural unification across cell signatures."""

    @staticmethod
    def inject_parameters(code_template: str, parameters: list[str], context=None) -> str:
        """
        Universally injects parameters into a function call code template using AST.
        e.g. output_var = cv2.cvtColor(input_var) -> output_var = cv2.cvtColor(input_var, cv2.COLOR_BGR2GRAY)
        """
        import ast
        try:
            tree = ast.parse(code_template)
            pos_params = [p for p in parameters if "=" not in p]
            kw_params = [p for p in parameters if "=" in p]
            ordered_params = pos_params + kw_params

            explicit_fn = None
            out_fn = None
            if context and hasattr(context, "extracted_parameters") and isinstance(context.extracted_parameters, dict):
                explicit_fn = context.extracted_parameters.get("explicit_filename") or context.extracted_parameters.get("input_filename")
                out_fn = context.extracted_parameters.get("output_filename")

            cell_stage = getattr(context.target_cell, 'stage', 0) if context and hasattr(context, 'target_cell') else 0

            # Traverse to find the first function call (ast.Call)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id.lower()
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr.lower()

                    # Classify cell I/O roles using node stage metadata rather than substring matching
                    if cell_stage == 1:  # Source / Reader Node
                        if explicit_fn and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is None:
                            node.args[0] = ast.Constant(value=explicit_fn)

                    if cell_stage == 3:  # Sink / Writer Node
                        if out_fn:
                            if not node.args:
                                node.args.append(ast.Constant(value=out_fn))
                            elif isinstance(node.args[0], ast.Constant) and (node.args[0].value is None or node.args[0].value == ''):
                                node.args[0] = ast.Constant(value=out_fn)

                    if func_name == "sort_values":
                        if not node.args and not any(kw.arg == "by" for kw in node.keywords):
                            by_col = None
                            if context and hasattr(context, "extracted_parameters") and isinstance(context.extracted_parameters, dict):
                                by_col = context.extracted_parameters.get("by") or context.extracted_parameters.get("sort_by")
                            if not by_col:
                                by_col = "{input_var}.columns[0]"
                            node.args.append(ast.parse(by_col, mode='eval').body)
                        is_descending = False
                        if context and hasattr(context, "extracted_parameters") and isinstance(context.extracted_parameters, dict):
                            is_descending = bool(context.extracted_parameters.get("descending"))
                        if not is_descending and context:
                            p_text = str(getattr(context, 'prompt_hint', '') or getattr(context, 'prompt', '') or getattr(context, 'user_prompt', '') or '').lower()
                            if re.search(r"\b(descending|desc|reverse)\b", p_text):
                                is_descending = True
                        if is_descending:
                            if not any(kw.arg == "ascending" for kw in node.keywords):
                                node.keywords.append(ast.keyword(arg="ascending", value=ast.Constant(value=False)))

                    existing_kwargs = {kw.arg for kw in node.keywords if kw.arg}
                    for p in ordered_params:
                        p_str = str(p).lower()
                        p_tokens = set(re.findall(r"[a-zA-Z0-9]+", p_str)) - {"true", "false", "none", "the", "a", "an", "and", "or", "in", "to", "from", "for", "with", "by", "is", "it"}
                        if not p_tokens:
                            continue
                        func_tokens = set(re.findall(r"[a-zA-Z0-9]+", func_name)) if func_name else set()
                        
                        # Generic AST parameter relevance validation via sub-token similarity & semantic roles
                        if p_tokens and func_tokens:
                            from difflib import SequenceMatcher
                            has_match = False

                            # Use cell parameter schemas to validate parameter relevance
                            has_match = False
                            if context and hasattr(context, 'target_cell') and context.target_cell:
                                for param in getattr(context.target_cell, 'parameters', []):
                                    if "=" in p_str:
                                        kw_key = p_str.split("=")[0].strip()
                                        if param.name == kw_key:
                                            has_match = True
                                            break
                                    elif param.name in p_str or str(param.default_value) in p_str:
                                        has_match = True
                                        break
                                        
                            if not has_match:
                                # Fallback to sub-token matching if cell schema is missing
                                for pt in p_tokens:
                                    if len(pt) <= 1:
                                        continue
                                    if any(pt in ft or ft in pt or SequenceMatcher(None, pt, ft).ratio() >= 0.55 for ft in func_tokens):
                                        has_match = True
                                        break

                            if not has_match:
                                continue

                        try:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                val_node = ast.parse(v.strip(), mode='eval').body
                                if k.strip() not in existing_kwargs:
                                    node.keywords.append(ast.keyword(arg=k.strip(), value=val_node))
                                    existing_kwargs.add(k.strip())
                            else:
                                arg_node = ast.parse(p.strip(), mode='eval').body
                                arg_repr = ast.unparse(arg_node)
                                existing_args = {ast.unparse(a) for a in node.args}
                                if arg_repr not in existing_args:
                                    node.args.append(arg_node)
                                    existing_args.add(arg_repr)
                        except Exception as inner_e:
                            logger.warning(f"[AST INJECTION WARNING] Could not parse parameter '{p}': {inner_e}")
                    break
            return ast.unparse(tree)
        except Exception as e:
            logger.error(f"[AST INJECTION ERROR] Failed to manipulate code template: {e}")
            return code_template

    @staticmethod
    def unify(context: ExecutionContext, target_cell, injected_parameters: list[str] = None) -> str:
        if context:
            context.target_cell = target_cell
        in_fname = context.extracted_parameters.get("input_filename") if context and context.extracted_parameters else None
        out_fname = context.extracted_parameters.get("output_filename") if context and context.extracted_parameters else None

        matching_input_var = context.find_compatible_variable(target_cell.inputs)

        if not matching_input_var:
            if context.registry:
                matching_input_var = list(context.registry.keys())[-1]
            else:
                matching_input_var = "input_source"

        cell_id = getattr(target_cell, "cell_id", "") or ""
        if cell_id and cell_id != "SYNTHESIZED_NODE":
            parts = cell_id.lower().split('_')
            if len(parts) > 1 and parts[0] in ["pandas", "opencv", "scikit"]:
                parts = parts[1:]
            raw_output_name = "_".join(parts)
        else:
            raw_output_name = getattr(target_cell.outputs, "state", "").lower().strip() or "output_var"
            if raw_output_name == "computed":
                raw_output_name = "computed_var"
        
        output_var_name = context.declare_variable(
            name=raw_output_name,
            signature=target_cell.outputs,
        )
        # Collect declared dependencies
        cell_deps = getattr(target_cell, "dependencies", []) or []
        if context and hasattr(context, "declared_dependencies"):
            if isinstance(cell_deps, (list, tuple, set)):
                context.declared_dependencies.update(cell_deps)
            elif isinstance(cell_deps, str):
                context.declared_dependencies.add(cell_deps)

        compiled_snippet = getattr(target_cell, "code_template", None) or getattr(target_cell, "code", None) or getattr(target_cell, "_code_template", "") or ""
        if not compiled_snippet and hasattr(target_cell, "domain_implementations") and isinstance(target_cell.domain_implementations, dict):
            py_impl = target_cell.domain_implementations.get("Python_Core", {})
            if isinstance(py_impl, dict):
                compiled_snippet = py_impl.get("code", "")

        
        # 0. Deterministic schema-driven slot resolution & pre-flight verification gate
        if compiled_snippet:
            prompt_hint = context.prompt_hint if (context and hasattr(context, 'prompt_hint') and context.prompt_hint) else ""
            extracted_params = ParameterExtractor.extract_parameters(prompt_hint)
            if context and hasattr(context, "extracted_parameters") and context.extracted_parameters:
                for k, v in context.extracted_parameters.items():
                    if v and not extracted_params.get(k):
                        extracted_params[k] = v

            slot_bindings = resolve_node_slots(compiled_snippet, extracted_params=extracted_params, context=context, target_cell=target_cell)
            for slot_k, slot_v in slot_bindings.items():
                compiled_snippet = compiled_snippet.replace("{" + slot_k + "}", slot_v)

            all_bindings = dict(context.extracted_parameters) if context and hasattr(context, 'extracted_parameters') and context.extracted_parameters else {}
            all_bindings.update(slot_bindings)
            assert_placeholders_resolved(compiled_snippet, bindings=all_bindings)

        # 1. Universal template-driven placeholder replacement
        if compiled_snippet:

            # Ensure write/save cells receive out_fname as 1st argument if template only had {input_var} or ()
            if out_fname:

                compiled_snippet = compiled_snippet.replace("{output_filename}", repr(out_fname))
                is_sink = getattr(target_cell.outputs, "type_name", "") == "None" or getattr(target_cell, "metadata_tags", {}).get("is_sink", False)
                if is_sink and repr(out_fname) not in compiled_snippet:
                    if "({input_var})" in compiled_snippet:
                        compiled_snippet = compiled_snippet.replace("({input_var})", f"({repr(out_fname)}, {{input_var}})")
                    elif "()" in compiled_snippet:
                        compiled_snippet = compiled_snippet.replace("()", f"({repr(out_fname)})")

            compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
            compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

            if in_fname:
                compiled_snippet = compiled_snippet.replace("{input_filename}", repr(in_fname))
                compiled_snippet = compiled_snippet.replace("{input_source}", repr(in_fname))
            else:
                compiled_snippet = compiled_snippet.replace("{input_filename}", matching_input_var)
                compiled_snippet = compiled_snippet.replace("{input_source}", matching_input_var)

        # 2. Dynamic heuristic parameter injection via AST
        prompt_heuristics = context.extracted_parameters.get("heuristics", []) if context and context.extracted_parameters else []
        cell_matched = getattr(target_cell, "matched_heuristics", []) or []
        cell_heuristics = list(dict.fromkeys(list(cell_matched) + list(prompt_heuristics)))
        if cell_heuristics and compiled_snippet:
            # Filter out filenames from heuristics so they aren't injected twice
            injected_params = [
                h for h in cell_heuristics
                if h != repr(in_fname) and h != repr(out_fname) and h != in_fname and h != out_fname
            ]
            if injected_params:
                compiled_snippet = UnificationGate.inject_parameters(compiled_snippet, injected_params, context=context)
        
        # If write/export call has no filename argument, inject output_filename if available
        if out_fname and compiled_snippet:
            try:
                tree = ast.parse(compiled_snippet)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func_name = node.func.attr.lower() if isinstance(node.func, ast.Attribute) else (node.func.id.lower() if isinstance(node.func, ast.Name) else "")
                        is_sink = getattr(target_cell.outputs, "type_name", "") == "None" or getattr(target_cell, "metadata_tags", {}).get("is_sink", False)
                        if is_sink and not node.args:
                            compiled_snippet = UnificationGate.inject_parameters(compiled_snippet, [repr(out_fname)], context=context)
            except Exception:
                pass

        # AST Lineage Repair: Ensure transformed variables in context are properly consumed by downstream calls
        compiled_snippet = UnificationGate.fix_dead_variables_in_snippet(context, compiled_snippet, current_output_var=output_var_name)

        # If the snippet defines a function but no longer assigns the output variable
        # (e.g., after stripping a trailing call with unresolved args), bind the
        # output variable to the function name so downstream cells can reference it.
        if compiled_snippet and output_var_name not in compiled_snippet:
            try:
                tree = ast.parse(compiled_snippet)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        compiled_snippet += f"\n{output_var_name} = {node.name}"
                        break
            except Exception:
                pass

        logger.info(
            f"[UNIFICATION SUCCESS] Linked {matching_input_var} -> {cell_id} -> {output_var_name} | Code: {compiled_snippet.strip()}"
        )
        return compiled_snippet

    @staticmethod
    def fix_dead_variables_in_snippet(context: ExecutionContext, snippet: str, current_output_var: str = None) -> str:
        """Fixes dead transformed variables by rebinding unmapped caller variables to the latest active variable in context."""
        if not context.registry:
            return snippet
        valid_vars = set(v for v in context.registry.keys() if v != current_output_var and v != "input_source")
        latest_var = list(context.registry.keys())[-1]

        try:
            tree = ast.parse(snippet)

            is_func_def = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            func_param_names = set()
            defined_func_names = set()
            if is_func_def:
                for n in ast.walk(tree):
                    if isinstance(n, ast.FunctionDef):
                        for arg in n.args.args:
                            func_param_names.add(arg.arg)
                        defined_func_names.add(n.name)

                if isinstance(tree, ast.Module) and defined_func_names:
                    cleaned_body = []
                    for stmt in tree.body:
                        should_strip = False
                        call_node = None
                        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                            call_node = stmt.value
                        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            call_node = stmt.value
                        if call_node and isinstance(call_node.func, ast.Name) and call_node.func.id in defined_func_names:
                            for arg in call_node.args:
                                if isinstance(arg, ast.Name) and arg.id in func_param_names:
                                    should_strip = True
                                    break
                        if not should_strip:
                            cleaned_body.append(stmt)
                    tree.body = cleaned_body if cleaned_body else tree.body

            snippet_local_vars = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            snippet_local_vars.add(target.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        snippet_local_vars.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        snippet_local_vars.add(alias.asname or alias.name)

            class CallerRebinder(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        caller = node.func.value.id
                        if (not is_module_available(caller) and caller not in valid_vars
                                and caller not in snippet_local_vars
                                and caller != "input_source" and caller != current_output_var
                                and caller not in func_param_names):
                            if not is_func_def and valid_vars:
                                logger.info(f"[AST LINEAGE REPAIR] Rebound unbound caller variable '{caller}' -> '{latest_var}' for method .{node.func.attr}()")
                                node.func.value.id = latest_var
                    if not is_func_def:
                        new_args = []
                        for arg in node.args:
                            if isinstance(arg, ast.Name):
                                arg_id = arg.id
                                if (not is_module_available(arg_id) and arg_id not in valid_vars
                                        and arg_id not in snippet_local_vars
                                        and arg_id != "input_source" and arg_id != current_output_var
                                        and arg_id not in context.registry):
                                    if valid_vars:
                                        arg.id = list(valid_vars)[0]
                                    else:
                                        return ast.copy_location(ast.Constant(value=arg_id), arg)
                            new_args.append(arg)
                        node.args = new_args
                    return node

            tree = CallerRebinder().visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except Exception as e:
            logger.error(f"[AST LINEAGE REPAIR] Failed: {e}")
            return snippet

    @staticmethod
    def validate_synthesis(
        synthesized_dict: dict,
        expected_inputs: str,
        expected_outputs: str,
        trees_dir: str = "trees",
    ) -> bool:
        """
        Validates the synthesized MicroCell JSON against the required typestates.
        If valid, caches it permanently.
        """
        import json

        inputs = synthesized_dict.get("inputs", {})
        outputs = synthesized_dict.get("outputs", {})

        if not isinstance(inputs, dict):
            inputs = {}
        if not isinstance(outputs, dict):
            outputs = {}

        in_type  = inputs.get("type_name",  inputs.get("input_type",  ""))
        out_type = outputs.get("type_name", outputs.get("output_type", ""))

        if not types_unify(expected_inputs, in_type) or not types_unify(expected_outputs, out_type):
            logger.error(f"[UNIFICATION ERROR] Synthesized typestates do not unify. Expected {expected_inputs}->{expected_outputs}, got {in_type}->{out_type}")
            return False

        cache_dir  = os.path.join(trees_dir, "micro")
        cache_path = os.path.join(cache_dir, "synthesized_nodes.json")
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"domain_name": "Synthesized_Domain", "cells": []}
        else:
            data = {"domain_name": "Synthesized_Domain", "cells": []}

        new_cell_id = synthesized_dict.get("cell_id")
        if new_cell_id:
            data["cells"] = [
                cell for cell in data.get("cells", [])
                if not isinstance(cell, dict) or cell.get("cell_id") != new_cell_id
            ]
        data.setdefault("cells", []).append(synthesized_dict)

        import tempfile
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.error(f"[UNIFICATION CACHE ERROR] Failed to save to {cache_path}: {e}")
            try:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

        logger.info(f"[UNIFICATION CACHE] Successfully saved {synthesized_dict.get('cell_id')} to {cache_path}")
        return True

    @staticmethod
    def resolve_imports(code_text: str, context: 'ExecutionContext' = None, chain_nodes: List[Any] = None) -> str:
        """
        Smart Top-of-File Import Injection System.
        Collects required main and optional tree/node imports across the execution route,
        deduplicates aliased import statements, strips scattered inline import lines,
        and presents a clean, canonical PEP 8 import block at the very top of the generated code.
        """
        # Step 1: Strip existing import lines from cell code bodies to avoid inline duplicate scattering
        body_lines = []
        for line in code_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            body_lines.append(line)
        clean_code_body = "\n".join(body_lines).strip()

        # Step 2: Collect tree domains and node dependencies
        domains = set()
        declared_deps = set()

        if context and hasattr(context, "declared_dependencies"):
            declared_deps.update(context.declared_dependencies)

        if chain_nodes:
            for node in chain_nodes:
                domain_name = getattr(node, "domain", getattr(node, "domain_name", None))
                if domain_name:
                    domains.add(domain_name.lower())
                deps = getattr(node, "dependencies", []) or []
                if isinstance(deps, (list, tuple, set)):
                    declared_deps.update(deps)
                elif isinstance(deps, str):
                    declared_deps.add(deps)

        # Infer domains from code body if chain_nodes not supplied
        if re.search(r"\bpd\.", code_text) or "pandas" in code_text:
            domains.add("pandas")
        if re.search(r"\bnp\.", code_text) or "numpy" in code_text:
            domains.add("numpy")
        if re.search(r"\bcv2\.", code_text) or "cv2" in code_text:
            domains.add("cv2")
        if re.search(r"\bscipy\.", code_text) or "scipy" in code_text:
            domains.add("scipy")
        if "sklearn" in code_text or "StandardScaler" in code_text or "RandomForestClassifier" in code_text:
            domains.add("sklearn")

        stdlib_imports = set()
        third_party_imports = set()

        # Step 3: Main Tree Imports
        if "pandas" in domains or "pd" in declared_deps:
            third_party_imports.add("import pandas as pd")
        if "numpy" in domains or "np" in declared_deps:
            third_party_imports.add("import numpy as np")
        if "cv2" in domains or "opencv" in domains:
            third_party_imports.add("import cv2")
        if "scipy" in domains:
            third_party_imports.add("import scipy")
        if "sklearn" in domains:
            third_party_imports.add("import sklearn")

        # Step 4: Optional Node Symbol Imports
        symbol_optional_map = {
            "StandardScaler": "from sklearn.preprocessing import StandardScaler",
            "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler",
            "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
            "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
            "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
            "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
            "LinearRegression": "from sklearn.linear_model import LinearRegression",
            "SVC": "from sklearn.svm import SVC",
            "SVR": "from sklearn.svm import SVR",
            "train_test_split": "from sklearn.model_selection import train_test_split",
            "accuracy_score": "from sklearn.metrics import accuracy_score",
            "mean_squared_error": "from sklearn.metrics import mean_squared_error",
        }

        for symbol, import_stmt in symbol_optional_map.items():
            if re.search(r"\b" + re.escape(symbol) + r"\b", code_text):
                third_party_imports.add(import_stmt)

        # Standard Library Imports
        std_modules = ["heapq", "json", "math", "re", "os", "sys", "time", "random", "itertools", "functools"]
        for std_mod in std_modules:
            if re.search(r"\b" + std_mod + r"\b", code_text) and is_module_available(std_mod):
                stdlib_imports.add(f"import {std_mod}")

        # Assemble clean PEP 8 header
        header_sections = []
        if stdlib_imports:
            header_sections.append("\n".join(sorted(stdlib_imports)))
        if third_party_imports:
            header_sections.append("\n".join(sorted(third_party_imports)))

        if header_sections:
            full_header = "\n\n".join(header_sections)
            return f"{full_header}\n\n{clean_code_body}"
        return clean_code_body


class DataflowLineageTracker(ast.NodeTransformer):
    """
    Tracks sequential variable transformations and re-links sink calls
    to the latest valid descendant in the lineage chain.
    """
    def __init__(self, target_cells=None):
        self.target_cells = target_cells or []
        self.lineage_tree: Dict[str, str] = {}  # new_var -> parent_var
        self.latest_descendant: Dict[str, str] = {}  # root/ancestor -> newest_leaf
        self.assigned_vars: Set[str] = set()
        self._imported_names: Set[str] = set()  # module/import names that must not be lineage-tracked

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name.split('.')[0])
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name)
        if node.module:
            self._imported_names.add(node.module.split('.')[0])
        return node

    def visit_Assign(self, node: ast.Assign):
        # Re-bind sink calls BEFORE updating assigned_vars/latest_descendant
        # to prevent self-referencing uninitialized assignments
        self._rebind_sink_call(node.value)

        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            self.assigned_vars.add(target_name)

            parent_var = None
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
                    candidate = node.value.func.value.id
                    # Don't track module calls (e.g. cv2.imread) as data lineage
                    if candidate not in self._imported_names:
                        parent_var = candidate
                elif isinstance(node.value.func, ast.Name):
                    for arg in node.value.args:
                        if isinstance(arg, ast.Name) and arg.id in self.assigned_vars:
                            parent_var = arg.id
                            break

            if parent_var:
                root = self._get_root(parent_var)
                self.lineage_tree[target_name] = parent_var
                self.latest_descendant[root] = target_name
                self.latest_descendant[parent_var] = target_name

        return node

    def visit_Expr(self, node: ast.Expr):
        """Re-bind object invocations on terminal sinks."""
        self.generic_visit(node)
        self._rebind_sink_call(node.value)
        return node

    def _rebind_sink_call(self, call_node):
        if not isinstance(call_node, ast.Call):
            return

        method_name = ""
        callee_var = None
        if isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):
                callee_var = call_node.func.value.id
        elif isinstance(call_node.func, ast.Name):
            method_name = call_node.func.id

        is_sink = any(k in method_name.lower() for k in ["save", "write", "dump", "export", "to_"])
        for cell in self.target_cells:
            if getattr(cell.outputs, "type_name", "") == "None" or getattr(cell, "metadata_tags", {}).get("is_sink", False):
                is_sink = True
                break

        if is_sink:
            # Rebind callee object if it has a newer descendant (e.g. df.to_csv -> df_sorted.to_csv)
            # but NEVER rebind module names (e.g. cv2.imwrite must stay cv2.imwrite)
            if callee_var and callee_var in self.latest_descendant and callee_var not in self._imported_names:
                newest_var = self.latest_descendant[callee_var]
                if newest_var != callee_var:
                    call_node.func.value.id = newest_var

            # Rebind positional argument variables if they have a newer descendant (e.g. cv2.imwrite('out.jpg', img) -> img_gray)
            for arg in call_node.args:
                if isinstance(arg, ast.Name) and arg.id in self.latest_descendant:
                    arg.id = self.latest_descendant[arg.id]

    def _get_root(self, var_name: str) -> str:
        curr = var_name
        while curr in self.lineage_tree:
            curr = self.lineage_tree[curr]
        return curr


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    """Parses generated code, traces lineage, and auto-corrects stale variable usages."""
    try:
        tree = ast.parse(code)
        transformer = DataflowLineageTracker(target_cells=target_cells)
        corrected_tree = transformer.visit(tree)
        ast.fix_missing_locations(corrected_tree)
        return ast.unparse(corrected_tree)
    except Exception:
        return code


KNOWN_FILE_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".parquet", ".feather", ".xlsx", 
    ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".h5", ".pkl", ".sqlite", ".db"
}



class ParameterExtractor:
    @staticmethod
    def extract_slots(prompt: str) -> ExtractedSlots:
        slots = ExtractedSlots()
        if not prompt:
            return slots

        # 1. Extract file paths with extensions using exact word boundaries
        file_matches = list(dict.fromkeys(re.findall(r'[\'"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[\'"]?', prompt)))
        save_kw_pos = -1
        for kw in ["save", "to", "write", "output", "export", "destination"]:
            pos = prompt.lower().find(kw)
            if pos != -1 and (save_kw_pos == -1 or pos < save_kw_pos):
                save_kw_pos = pos

        for match in file_matches:
            _, ext = os.path.splitext(match)
            if ext.lower() in KNOWN_FILE_EXTENSIONS:
                m = re.search(r'\b' + re.escape(match) + r'\b', prompt)
                pos = m.start() if m else -1
                if save_kw_pos != -1 and pos > save_kw_pos:
                    slots.dest_uris.append(match)
                else:
                    slots.source_uris.append(match)

        # 2. Extract column/feature names guided by syntax markers
        col_matches = re.findall(
            r'(?:column|col|by|sort|filter|select|field|attribute|feature)\s+(?:on\s+|the\s+|a\s+|an\s+)*[\'"]([a-zA-Z0-9_\-\s]+)[\'"]', 
            prompt, 
            re.IGNORECASE
        )
        col_matches.extend(re.findall(
            r'[\'"]([a-zA-Z0-9_\-\s]+)[\'"]\s+(?:column|col|field|attribute|feature)',
            prompt,
            re.IGNORECASE
        ))
        col_unquoted = re.findall(
            r'(?:column|col|by|sort|filter|select|field|attribute|feature)\s+(?:on\s+|the\s+|a\s+|an\s+)*\b([a-zA-Z_][a-zA-Z0-9_]*)\b',
            prompt,
            re.IGNORECASE
        )
        col_unquoted = [c for c in col_unquoted if c.lower() not in {"descending", "ascending", "value", "values", "column", "columns", "data", "file", "csv", "the", "a", "an", "and", "or", "in", "to", "from", "for", "with"}]
        col_matches.extend(col_unquoted)
        slots.named_identifiers = list(dict.fromkeys(col_matches))

        # 3. Extract operational flags
        prompt_lower = prompt.lower()
        if re.search(r"\b(descending|desc|reverse|highest\s+to\s+lowest)\b", prompt_lower):
            slots.operational_flags["descending"] = True
        elif re.search(r"\b(ascending|asc|lowest\s+to\s+highest)\b", prompt_lower):
            slots.operational_flags["descending"] = False

        if re.search(r"\b(gray|grayscale|bgr2gray|rgb2gray)\b", prompt_lower):
            slots.operational_flags["color_space"] = "GRAY"
            slots.operational_flags["is_grayscale"] = True
        elif re.search(r"\b(hsv|bgr2hsv|rgb2hsv)\b", prompt_lower):
            slots.operational_flags["color_space"] = "HSV"
            slots.operational_flags["is_hsv"] = True
        elif re.search(r"\b(rgb|bgr2rgb)\b", prompt_lower):
            slots.operational_flags["color_space"] = "RGB"
            slots.operational_flags["is_rgb"] = True

        return slots

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()



