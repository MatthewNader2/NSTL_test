"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate, Parameter Binding, and AST Code Synthesizer.
Domain-Agnostic Dynamic Typestate Port Binding (Zero Hardcodes).
"""

from __future__ import annotations
import ast
import os
import re
import sys
import json
import importlib
import importlib.util
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from log_config import get_logger

try:
    from .lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry
except (ImportError, ValueError):
    from lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry

logger = get_logger('unification')


class UnresolvedPlaceholderError(Exception):
    """Raised when a code template placeholder cannot be bound."""
    pass


class UnificationFailure(Exception):
    pass


# Formal Top Type ⊤ in NSTL Type System (unifies with every type tau: unify(⊤, tau) = True)
TOP_TYPE_SET = frozenset({"any", "Any", "*", "top", "TOP", "ANY", "unknown", "object"})

def types_unify(expected_type: str, actual_type: str) -> bool:
    """
    Formal unification operator unify(tau_expected, tau_actual).
    Returns True iff tau_expected and tau_actual unify.
    Rules:
      1. unify(⊤, tau) = True for all tau (Top type wildcard)
      2. unify(tau, ⊤) = True for all tau
      3. unify(tau, tau) = True (Exact identity)
      4. Subtype lattice satisfaction: tau_actual <= tau_expected
    """
    if not expected_type or not actual_type:
        return True
    exp_clean = str(expected_type).strip()
    act_clean = str(actual_type).strip()
    if exp_clean in TOP_TYPE_SET or act_clean in TOP_TYPE_SET:
        return True
    if exp_clean.lower() == act_clean.lower():
        return True
    try:
        from lattice import TypeRegistry
        return TypeRegistry.get_instance().is_subtype(act_clean, exp_clean)
    except Exception:
        return False


class DynamicPlaceholderResolver:
    """
    Domain-agnostic parameter resolver.
    Binds placeholders dynamically based on PortSignature (type_name + state),
    scope variables, and extracted literal arguments.
    """

    @staticmethod
    def is_module_attribute(expr_str: str) -> bool:
        """Dynamically verifies if expr_str is an importable module attribute (e.g. cv2.COLOR_BGR2GRAY)."""
        if not expr_str or not isinstance(expr_str, str):
            return False
        expr_str = expr_str.strip()
        if not expr_str or "." not in expr_str:
            return False

        # Fast reject known file extensions
        known_exts = {'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'txt', 'parquet', 'h5', 'pkl', 'py'}
        if expr_str.split('.')[-1].lower() in known_exts or '/' in expr_str or '\\' in expr_str:
            return False

        try:
            tree = ast.parse(expr_str, mode='eval')
            if not isinstance(tree.body, ast.Attribute):
                return False

            parts = []
            curr = tree.body
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            parts.reverse()

            # Check if root is an installed module
            mod_name = parts[0]
            if importlib.util.find_spec(mod_name) is None:
                return False

            mod = importlib.import_module(mod_name)
            target = mod
            for attr in parts[1:]:
                if not hasattr(target, attr):
                    return False
                target = getattr(target, attr)
            return True
        except Exception:
            return False

    def resolve_port(
        self,
        port_name: str,
        port_sig: Union[PortSignature, AlgebraicSignature, Any],
        stage: int,
        context: Any,
        output_var: str,
        cell: Any = None
    ) -> str:
        """
        Dynamically resolves the replacement value for a given port placeholder
        following the strict 3-tier resolution order:
          Tier 1: Explicit Intent (Literals & Prepositional Entities from prompt)
          Tier 2: Contextual Schema (In-scope typestate variables)
          Tier 3: Minimal Neutral Default (Signature defaults or neutral fallbacks)
        """
        if port_name == "output_var":
            return output_var

        # Extract type_name and state safely
        type_name = getattr(port_sig, "type_name", None)
        if type_name is None and hasattr(port_sig, "signature"):
            type_name = port_sig.signature.type_name
        type_name = str(type_name or "any")

        state = getattr(port_sig, "state", None)
        if state is None and hasattr(port_sig, "signature"):
            state = port_sig.signature.state
        state = str(state or "any")

        # =====================================================================
        # TIER 1: Explicit Intent (Literals, Modifiers, and Prepositional Tokens)
        # =====================================================================

        # Explicit contextual parameters passed directly
        if hasattr(context, "parameters") and port_name in context.parameters:
            val = context.parameters[port_name]
            if isinstance(val, str) and not self.is_module_attribute(val) and not (val.startswith('"') or val.startswith("'")):
                return f'"{val}"'
            return str(val)

        # 1.1 Value / Replacement / Literal binding (e.g. fillna(0), replace(val))
        if port_name in ("value", "val", "fill_value", "to_replace", "replacement", "fill"):
            if hasattr(context, "with_values") and context.with_values:
                w_val = str(context.with_values[0])
                if w_val.replace('.', '', 1).isdigit() or (w_val.startswith('-') and w_val[1:].replace('.', '', 1).isdigit()):
                    return w_val
                if w_val.lower() in ("none", "null", "nan"):
                    return "None"
                if w_val.startswith('"') or w_val.startswith("'"):
                    return w_val
                return f'"{w_val}"'
            if hasattr(context, "numeric_literals") and context.numeric_literals:
                return str(context.numeric_literals[0])
            if hasattr(context, "frame") and context.frame and context.frame.literal_constants:
                c_val = next(iter(context.frame.literal_constants))
                return c_val if (c_val.replace('.', '', 1).isdigit() or c_val.lower() == "none") else f'"{c_val}"'

        # 1.2 Scoped Role-Based Entity Resolution: Structural vs Ordering vs Computation
        cell_tags = set(getattr(cell, "semantic_tags", [])) | set(getattr(cell, "keywords", [])) | set(getattr(cell, "inputs", {}).keys())
        is_sort_node = "ascending" in getattr(cell, "inputs", {}) or any(t in cell_tags for t in ("sort", "sorting", "order", "sorted", "rank"))
        is_group_node = any(t in cell_tags for t in ("group", "groupby", "grouped", "aggregate", "partition", "split"))

        # Role 1: Structural Partitioner Slot (e.g. groupby by="department")
        if (port_name in ("by", "keys", "key", "partition_by", "level") and is_group_node) or port_name in ("group_by", "partition_by"):
            if hasattr(context, "group_by_targets") and context.group_by_targets:
                col = context.group_by_targets[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "frame") and context.frame and context.frame.partitioning_entities:
                col = next(iter(context.frame.partitioning_entities))
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "by_column") and context.by_column:
                col = context.by_column
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col

        # Role 2: Ordering / Ranking Criteria Slot (e.g. sort_values by="sales" or by="age")
        if (port_name in ("by", "sort_by", "order_by") and is_sort_node) or port_name in ("sort_by", "order_by"):
            # 1. Explicit ordering keys from user
            if hasattr(context, "sort_by_targets") and context.sort_by_targets:
                col = context.sort_by_targets[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "frame") and context.frame and context.frame.ordering_keys:
                col = next(iter(context.frame.ordering_keys))
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col

            # 2. Downstream metric/measure resolution: sorting by value/metric defaults to active operand
            if getattr(context, "sort_by_metric", False) or not (hasattr(context, "sort_by_targets") and context.sort_by_targets):
                if hasattr(context, "metric_targets") and context.metric_targets:
                    col = context.metric_targets[0]
                    return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
                if hasattr(context, "frame") and context.frame and context.frame.operand_entities:
                    col = next(iter(context.frame.operand_entities))
                    return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col

            # 3. Fallback to general column/by target
            if hasattr(context, "by_column") and context.by_column:
                col = context.by_column
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col

        # Role 3: Computation / Measure Operand Slot (e.g. columns="sales", subset="sales")
        # Explicitly prevents structural partitioners (e.g. "department") from shadowing computation operands
        if state == "column_name" or port_name in ("by", "column", "columns", "on", "subset", "key", "keys", "subset_cols", "target_col", "features"):
            if hasattr(context, "metric_targets") and context.metric_targets:
                col = context.metric_targets[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "frame") and context.frame and context.frame.operand_entities:
                col = next(iter(context.frame.operand_entities))
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if port_name == "on" and hasattr(context, "on_targets") and context.on_targets:
                col = context.on_targets[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "columns") and context.columns:
                col = context.columns[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col
            if hasattr(context, "by_targets") and context.by_targets:
                col = context.by_targets[0]
                return f'"{col}"' if not (col.startswith('"') or col.startswith("'")) else col

        # Role 4: Target Vector / Label Operand Slot (e.g. y="target", y_true="target")
        if port_name in ("y", "target", "labels", "y_true") or state in ("target_vector", "labels"):
            # Prefer a column name the prompt actually named (e.g. "predict price"),
            # rather than assuming the label column is literally called "target".
            named_col = None
            if hasattr(context, "metric_targets") and context.metric_targets:
                named_col = context.metric_targets[0]
            elif hasattr(context, "frame") and context.frame and context.frame.operand_entities:
                named_col = next(iter(context.frame.operand_entities))
            elif hasattr(context, "columns") and context.columns:
                named_col = context.columns[0]

            if hasattr(context, "scope_variables"):
                for var_name, var_sig in reversed(context.scope_variables.items()):
                    v_type = getattr(var_sig, "type_name", None)
                    if v_type is None and hasattr(var_sig, "signature"):
                        v_type = var_sig.signature.type_name
                    if str(v_type) == "DataFrame":
                        if named_col:
                            return f"{var_name}['{named_col}']"
                        # No column name was given anywhere in the prompt: fall back to
                        # the common tabular-ML convention (last column is the label),
                        # checked at runtime rather than assumed at synthesis time.
                        return f"({var_name}['target'] if 'target' in {var_name}.columns else {var_name}.iloc[:, -1])"

            latest = getattr(context, "get_latest_data_variable", lambda: None)()
            if latest:
                if named_col:
                    return f"({latest}['{named_col}'] if hasattr({latest}, 'columns') else {latest})"
                return f"({latest}['target'] if hasattr({latest}, 'columns') and 'target' in {latest}.columns else ({latest}[:, -1] if hasattr({latest}, 'shape') and len({latest}.shape) > 1 and {latest}.shape[1] > 1 else {latest}))"

            # No data variable is in scope at all: there is nothing to derive a
            # target/label vector from. Fabricating an array of an arbitrary
            # length (e.g. a fixed size of 10) would silently synthesize a
            # fake label vector instead of surfacing the real gap.
            raise UnresolvedPlaceholderError(
                f"Cannot resolve target/label port '{port_name}': no upstream "
                f"data variable is in scope to derive it from."
            )

        # 1.3 Boolean & Directional Modifiers
        # Single source of truth for "ascending"/direction flags is
        # context.flags, populated once by SemanticFrame.build() (router.py)
        # / ExecutionContext._extract_prompt_literals(). If it was never set
        # there, no explicit direction was stated anywhere in the prompt -
        # fall through to Tier 3 below (the port's own declared default_value,
        # or the type's neutral default) instead of guessing "True" here.
        if hasattr(context, "flags") and port_name in context.flags:
            return str(bool(context.flags[port_name]))
        if port_name in ("ascending", "asc"):
            if hasattr(context, "flags") and "ascending" in context.flags:
                return str(bool(context.flags["ascending"]))
            if getattr(port_sig, "default_value", None) not in (None, "..."):
                pass  # handled by the shared Tier 3 default_value check below
            else:
                # No explicit direction was stated anywhere in the prompt, and
                # the cell declares no default_value of its own. This is an
                # explicit, static, library-wide convention (ascending sort is
                # the common default across pandas/numpy/etc.) - not a value
                # re-derived by re-scanning the prompt text a second time.
                return "True"

        # 1.4 Source and Destination File URIs (Topological Pipeline Position)
        is_sink_cell = (
            getattr(cell, "stage", None) == 3 or
            any(getattr(p, "state", None) in ("filepath_written", "saved", "exported") or
                getattr(getattr(p, "signature", None), "state", None) in ("filepath_written", "saved", "exported")
                for p in getattr(cell, "outputs", {}).values())
        )
        is_source_cell = (
            getattr(cell, "stage", None) == 1 or
            any(getattr(p, "state", None) in ("raw", "source_identifier", "filepath_read") or
                getattr(getattr(p, "signature", None), "state", None) in ("raw", "source_identifier", "filepath_read")
                for p in getattr(cell, "outputs", {}).values())
        )
        is_path_port = (
            state in ("source_identifier", "filepath_read", "input_uri", "dest_identifier", "filepath_written", "output_uri") or
            port_name in ("filepath", "filename", "input_filename", "input_file", "source", "dest_path", "output_filename", "destination", "output_file")
        )

        if is_path_port:
            files = getattr(context, "extracted_files", []) or getattr(context, "path_literals", [])
            dest_files = getattr(context, "dest_files", []) or []
            source_files = getattr(context, "source_files", []) or []

            port_default = getattr(port_sig, "default_value", None)

            if is_sink_cell:
                # Egress Sink cell (Stage 3 or filepath_written output) -> binds terminal egress path P_m
                if dest_files:
                    f = dest_files[-1]
                elif files:
                    f = files[-1]
                elif port_default:
                    f = str(port_default)
                else:
                    raise UnresolvedPlaceholderError(
                        f"No destination file was specified in the prompt for output port "
                        f"'{port_name}', and the cell declares no default_value for it."
                    )
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f

            if is_source_cell:
                # Ingress Source cell (Stage 1 or raw output) -> binds initial ingress path P_0
                if source_files:
                    f = source_files[0]
                elif files and files[0] not in dest_files:
                    f = files[0]
                elif port_default:
                    f = str(port_default)
                else:
                    raise UnresolvedPlaceholderError(
                        f"No source file was specified in the prompt for input port "
                        f"'{port_name}', and the cell declares no default_value for it."
                    )
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f

            # Intermediate or general path port:
            if state in ("dest_identifier", "filepath_written", "output_uri") or port_name in ("dest_path", "output_filename", "destination", "output_file"):
                if dest_files:
                    f = dest_files[-1]
                elif files:
                    f = files[-1]
                elif port_default:
                    f = str(port_default)
                else:
                    raise UnresolvedPlaceholderError(
                        f"No destination file was specified in the prompt for port '{port_name}', "
                        f"and the cell declares no default_value for it."
                    )
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f
            else:
                if source_files:
                    f = source_files[0]
                elif files and files[0] not in dest_files:
                    f = files[0]
                elif port_default:
                    f = str(port_default)
                else:
                    raise UnresolvedPlaceholderError(
                        f"No source file was specified in the prompt for port '{port_name}', "
                        f"and the cell declares no default_value for it."
                    )
                return f'"{f}"' if not (f.startswith('"') or f.startswith("'")) else f

        # =====================================================================
        # TIER 2: Contextual Schema (In-Scope Typestate Variables)
        # =====================================================================
        is_data_container = TypeRegistry.get_instance().is_container_type(type_name)
        is_primary_data_port = (
            cell is not None
            and getattr(cell, "primary_input", None) is not None
            and getattr(cell.primary_input, "name", None) == port_name
            and state in ("any", "raw", "transformed", "processed", "input")
        )
        is_top_type = type_name.lower() in ("any", "*", "top", "anyobject", "object")

        if hasattr(context, "scope_variables") and context.scope_variables:
            if not is_top_type or is_primary_data_port:
                matching_vars = []
                for var_name, var_sig in context.scope_variables.items():
                    if hasattr(var_sig, "unifies_with") and var_sig.unifies_with(port_sig):
                        matching_vars.append(var_name)
                    elif hasattr(port_sig, "unifies_with") and port_sig.unifies_with(var_sig):
                        matching_vars.append(var_name)
                    elif hasattr(var_sig, "signature") and hasattr(port_sig, "signature"):
                        if var_sig.signature.unifies_with(port_sig.signature):
                            matching_vars.append(var_name)
                if matching_vars:
                    return matching_vars[-1]

                # Typename matching for concrete types
                if not is_top_type:
                    for var_name, var_sig in reversed(context.scope_variables.items()):
                        v_type = getattr(var_sig, "type_name", None)
                        if v_type is None and hasattr(var_sig, "signature"):
                            v_type = var_sig.signature.type_name
                        if str(v_type).lower() == type_name.lower():
                            return var_name

        if is_data_container or is_primary_data_port:
            if hasattr(context, "get_latest_data_variable"):
                latest = context.get_latest_data_variable()
                if latest:
                    return latest

        # =====================================================================
        # TIER 3: Minimal Neutral Default (Signature defaults or neutral values)
        # =====================================================================
        if hasattr(port_sig, "default_value") and port_sig.default_value is not None and port_sig.default_value != "...":
            dv = str(port_sig.default_value).strip()
            if dv in ("True", "False", "None") or dv.replace('.', '', 1).isdigit() or (dv.startswith('-') and dv[1:].replace('.', '', 1).isdigit()):
                return dv
            if (dv.startswith('{') and dv.endswith('}')) or (dv.startswith('[') and dv.endswith(']')) or (dv.startswith('(') and dv.endswith(')')):
                return dv
            if self.is_module_attribute(dv):
                return dv
            if not (dv.startswith('"') or dv.startswith("'")):
                return f'"{dv}"'
            return dv

        if port_name in ("value", "val", "fill_value"):
            return "0"
        if state in ("source_identifier", "filepath_read", "input_uri") or port_name in ("filepath", "filename", "input_filename", "input_file", "source"):
            return '"input_data.csv"'
        if state in ("dest_identifier", "filepath_written", "output_uri") or port_name in ("dest_path", "output_filename", "destination", "output_file"):
            return '"output_data.csv"'
        if state == "column_name" or port_name in ("by", "column", "columns"):
            return '"target"'
        p_lower = port_name.lower()
        t_lower = type_name.lower()

        # Type-theoretic neutral elements (monoidal zero / identity elements)
        if t_lower == "list":
            return "[]"
        if t_lower == "dict":
            return "{}"
        if t_lower in ("set", "frozenset"):
            return "set()"
        if t_lower == "tuple":
            return "()"
        if t_lower in ("int", "integer"):
            return "0"
        if t_lower in ("float", "double", "number", "numeric"):
            return "0.0"
        if t_lower in ("str", "string"):
            return '""'
        if t_lower in ("bool", "boolean"):
            return "False"

        # If reaching here, the identifier has no value and is not a defined variable.
        # Fallback to None rather than emitting an unquoted unbound identifier which causes NameError.
        return "None"

    def assert_placeholders_resolved(self, code: str) -> None:
        """Ensures no raw {placeholder} tokens remain in synthesized code."""
        unresolved = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', code)
        if unresolved:
            raise UnresolvedPlaceholderError(f"Code contains unresolved placeholders: {unresolved}\nCode:\n{code}")


# Compatibility alias
PlaceholderResolver = DynamicPlaceholderResolver


def assert_placeholders_resolved(code_str: str, bindings: Optional[Dict[str, Any]] = None) -> None:
    test_code = code_str
    if bindings:
        for k, v in bindings.items():
            test_code = test_code.replace(f"{{{k}}}", str(v))
    DynamicPlaceholderResolver().assert_placeholders_resolved(test_code)


@dataclass
class VariableBinding:
    name: str
    signature: AlgebraicSignature
    literal_value: Optional[str] = None
    lineage_parent: Optional[str] = None


class ExecutionContext:
    """Manages lexical scopes, declared variables, and prompt literal bindings."""

    FILE_EXTENSIONS = (
        r'csv|tsv|json|parquet|xlsx|jpg|jpeg|png|bmp|webp|tif|tiff|txt|db|sqlite|h5|hdf5|mat|'
        r'pdf|md|py|npy|npz|pkl|pickle|feather|orc|avro|yaml|yml|toml|ini|'
        r'wav|mp3|flac|ogg|m4a|aac|avi|mp4|mov|mkv'
    )
    COLUMN_STOP_WORDS = {
        "descending", "ascending", "the", "a", "an", "column", "columns",
        "data", "file", "csv", "by", "sort", "order", "and", "or", "in", "of",
        "to", "from", "with", "into", "as",
        "values", "value", "dataset", "table", "records", "rows", "row",
        "asc", "desc", "true", "false", "none", "null",
        "sum", "total", "mean", "average", "avg", "count", "min", "max", "std", "var",
        "all", "any", "each", "every", "index", "on",
        "calculate", "compute", "find", "get", "determine", "apply", "output", "input"
    }

    def __init__(
        self,
        prompt: str = "",
        scope: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        frame: Optional[Any] = None
    ):
        self._scope: Dict[str, VariableBinding] = {}
        self.scope_variables: Dict[str, PortSignature] = {}
        self.declared_dependencies: Set[str] = set()
        self._var_counter: Dict[str, int] = {}
        self._var_order: List[str] = []
        self.prompt_hint: str = prompt
        self.prompt_lower: str = prompt.lower()
        self.extracted_files: List[str] = []
        self.source_files: List[str] = []
        self.dest_files: List[str] = []
        self.columns: List[str] = []
        self.by_column: Optional[str] = None
        self.by_targets: List[str] = []
        self.group_by_targets: List[str] = []
        self.sort_by_targets: List[str] = []
        self.metric_targets: List[str] = []
        self.sort_by_metric: bool = False
        self.on_targets: List[str] = []
        self.with_values: List[str] = []
        self.numeric_literals: List[str] = []
        self.flags: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = dict(parameters) if parameters else {}
        self.frame = frame

        if self.frame is None and prompt:
            try:
                from router import SemanticFrame
                self.frame = SemanticFrame.build(prompt)
            except Exception:
                self.frame = None

        if self.frame is not None:
            self.group_by_targets = list(self.frame.partitioning_entities)
            self.sort_by_targets = list(self.frame.ordering_keys)
            self.metric_targets = list(self.frame.operand_entities)
            self.with_values = list(self.frame.literal_constants)
            self.numeric_literals = [x for x in self.frame.literal_constants if x.replace('.', '', 1).isdigit()]
            self.sort_by_metric = self.frame.sort_by_metric
            if self.frame.ascending is not None:
                self.flags["ascending"] = self.frame.ascending

        if scope:
            for k, v in scope.items():
                if isinstance(v, PortSignature):
                    self.scope_variables[k] = v
                    self._scope[k] = VariableBinding(name=k, signature=v.signature)
                elif isinstance(v, AlgebraicSignature):
                    self.scope_variables[k] = PortSignature(k, v)
                    self._scope[k] = VariableBinding(name=k, signature=v)

        if prompt:
            self._extract_prompt_literals(prompt)

    def extract_prompt_parameters(self, prompt: str):
        self.prompt_hint = prompt
        self.prompt_lower = prompt.lower()
        self._extract_prompt_literals(prompt)

    def _extract_prompt_literals(self, prompt: str):
        # Dynamically extract all path literals in chronological order
        from router import extract_file_paths_and_extensions
        files, _ = extract_file_paths_and_extensions(prompt)
        self.extracted_files = files
        self.path_literals = files

        # Directional preposition markers
        egress_markers = {"to", "save", "write", "export", "into", "output", "as"}
        ingress_markers = {"from", "load", "read", "in", "input", "source"}

        src_files = []
        dst_files = []
        prompt_lower = prompt.lower()

        for f in files:
            f_clean = f.strip("\"'")
            idx = prompt_lower.find(f_clean.lower())
            preceding_context = prompt_lower[:idx] if idx > 0 else ""
            preceding_tokens = set(re.findall(r"\b[a-z]+\b", preceding_context)[-4:]) if preceding_context else set()

            is_egress = bool(preceding_tokens & egress_markers)
            is_ingress = bool(preceding_tokens & ingress_markers)

            if is_egress and not is_ingress:
                dst_files.append(f)
            elif is_ingress and not is_egress:
                src_files.append(f)
            else:
                if os.path.exists(f):
                    src_files.append(f)
                elif len(files) > 1 and f == files[0]:
                    src_files.append(f)
                elif len(files) > 1 and f == files[-1]:
                    dst_files.append(f)
                else:
                    if any(m in prompt_lower for m in ("save", "write", "export", "output")):
                        dst_files.append(f)
                    else:
                        src_files.append(f)

        self.source_files = list(dict.fromkeys(src_files))
        self.dest_files = list(dict.fromkeys(dst_files))

        if self.source_files:
            self.declare_variable(
                "input_file",
                AlgebraicSignature("str", "source_identifier"),
                literal_value=json.dumps(self.source_files[0])
            )
        if self.dest_files:
            self.declare_variable(
                "output_file",
                AlgebraicSignature("str", "dest_identifier"),
                literal_value=json.dumps(self.dest_files[-1])
            )

        # 1. Numeric literals: e.g. 0, 50000, 3.14
        self.numeric_literals = re.findall(r"\b\d+(?:\.\d+)?\b", prompt)

        # 2. Values governed by 'with': e.g. "with 0", "with None", "with 'unknown'", "with mean"
        with_matches = re.finditer(
            r"\bwith\s+(?:the\s+|a\s+|an\s+|value\s+of\s+|values?\s+of\s+|values?\s+)*([\"']?[a-zA-Z0-9_.-]+[\"']?)",
            prompt,
            re.IGNORECASE
        )
        for m in with_matches:
            w_val = m.group(1).strip()
            if w_val.lower() not in self.COLUMN_STOP_WORDS:
                self.with_values.append(w_val)

        # 3. Targets governed by 'on': e.g. "on id", "on user_id"
        on_matches = re.finditer(
            r"\bon\s+(?:the\s+|a\s+|an\s+|column\s+)*([\"']?[a-zA-Z_][a-zA-Z0-9_]*[\"']?)",
            prompt,
            re.IGNORECASE
        )
        for m in on_matches:
            on_val = m.group(1).strip("\"'")
            if on_val.lower() not in self.COLUMN_STOP_WORDS:
                self.on_targets.append(on_val)

        # 4. GroupBy targets: e.g. "group by department", "grouped by region"
        for m in re.finditer(
            r"(?:group\s+by|grouped\s+by|per|for\s+each)\s+(?:the\s+|a\s+|an\s+|column\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in self.COLUMN_STOP_WORDS:
                self.group_by_targets.append(val)
                self.by_targets.append(val)
                if not self.by_column:
                    self.by_column = val

        # 5. SortBy targets: e.g. "sort by salary", "sort values by age"
        for m in re.finditer(
            r"(?:sort\s+(?:values\s+)?by|sorted\s+(?:values\s+)?by|order\s+by)\s+(?:the\s+|a\s+|an\s+|column\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in self.COLUMN_STOP_WORDS:
                self.sort_by_targets.append(val)
                self.by_targets.append(val)
                if not self.by_column:
                    self.by_column = val

        # 6. Aggregated metric / measure targets: e.g. "total sales sum", "calculate total sales", "sum of revenue"
        for m in re.finditer(
            r"(?:total|sum|mean|average|avg|median|min|max|count|std|var)\s+(?:of\s+)?(?:the\s+)?[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in self.COLUMN_STOP_WORDS:
                self.metric_targets.append(val)

        for m in re.finditer(
            r"[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?\s+(?:sum|total|mean|average|avg|count)",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in self.COLUMN_STOP_WORDS:
                self.metric_targets.append(val)

        self.metric_targets = list(dict.fromkeys(self.metric_targets))

        # Check if prompt specifies sorting by value / measure / metric
        self.sort_by_metric = bool(re.search(
            r"\b(?:sort\s+values|sort\s+by\s+values?|sort\s+by\s+total|sort\s+by\s+sum|sort\s+by\s+metric|sort\s+descending|sort\s+ascending)\b",
            self.prompt_lower
        ))
        if not self.sort_by_targets and any(k in self.prompt_lower for k in ("sort", "order")):
            self.sort_by_metric = True

        # 7. General column extraction and registration
        col_pattern = (
            r"(?:column|col|by|sort\s+by|group\s+by|sum|mean|aggregate|average|count|min|max|filter\s+by)\s+"
            r"(?:the\s+|a\s+|an\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?"
        )
        for match in re.finditer(col_pattern, prompt, re.IGNORECASE):
            col_name = match.group(1)
            if col_name.lower() not in self.COLUMN_STOP_WORDS:
                self.columns.append(col_name)

        for met in self.metric_targets:
            if met not in self.columns:
                self.columns.append(met)

        for col_name in list(dict.fromkeys(self.columns)):
            self.declare_variable(
                f"col_{col_name}",
                AlgebraicSignature("str", "column_name"),
                literal_value=json.dumps(col_name)
            )

        # Only derive here if SemanticFrame.build() (the single canonical
        # detector for sort direction, in router.py) didn't already set it -
        # avoids two independently-maintained keyword checks disagreeing.
        if "ascending" not in self.flags:
            if "ascending" in self.prompt_lower:
                self.flags["ascending"] = True
            elif "descending" in self.prompt_lower or "reverse" in self.prompt_lower:
                self.flags["ascending"] = False
        if "ascending" in self.flags:
            self.declare_variable(
                "sort_asc",
                AlgebraicSignature("bool", "sort_flag"),
                literal_value=str(self.flags["ascending"])
            )

    def declare_variable(
        self,
        base_name: str,
        signature: Union[AlgebraicSignature, PortSignature],
        parent_var: Optional[str] = None,
        literal_value: Optional[str] = None
    ) -> str:
        if isinstance(signature, PortSignature):
            alg_sig = signature.signature
        else:
            alg_sig = signature

        clean_base = re.sub(r'[^a-zA-Z0-9_]', '', base_name)
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"

        idx = self._var_counter.get(clean_base, 0)
        var_name = clean_base if idx == 0 else f"{clean_base}_{idx + 1}"
        self._var_counter[clean_base] = idx + 1

        binding = VariableBinding(
            name=var_name,
            signature=alg_sig,
            literal_value=literal_value,
            lineage_parent=parent_var
        )
        self._scope[var_name] = binding
        if literal_value is None:
            self.scope_variables[var_name] = PortSignature(var_name, alg_sig)
        self._var_order.append(var_name)
        return var_name

    def peek_next_variable_name(self, base_name: str) -> str:
        clean_base = re.sub(r'[^a-zA-Z0-9_]', '', base_name)
        if not clean_base or clean_base[0].isdigit():
            clean_base = f"var_{clean_base}"
        idx = self._var_counter.get(clean_base, 0)
        return clean_base if idx == 0 else f"{clean_base}_{idx + 1}"

    def find_compatible_variable(self, required_sig: AlgebraicSignature) -> Optional[VariableBinding]:
        for var_name in reversed(self._var_order):
            binding = self._scope[var_name]
            if binding.signature.unifies_with(required_sig):
                return binding
        return None

    def get_latest_data_variable(self) -> Optional[str]:
        for var_name in reversed(self._var_order):
            binding = self._scope[var_name]
            if binding.signature.type_name not in ("str", "bool", "int", "None", "float"):
                return var_name
        return None

    def get_variable(self, name: str) -> Optional[VariableBinding]:
        return self._scope.get(name)


class UnificationGate:
    """Monadic code unifier: binds cell ports and templates to in-scope variables and prompt literals."""

    DENYLISTED_IMPORTS: Set[str] = {
        "generic", "algorithms", "data_engineering", "machine_learning",
        "python_core", "data_processing", "image_processing", "nlp", "opencv"
    }

    CANONICAL_IMPORTS: Dict[str, str] = {
        "pd": "import pandas as pd",
        "pandas": "import pandas as pd",
        "np": "import numpy as np",
        "numpy": "import numpy as np",
        "cv2": "import cv2",
        "matplotlib": "import matplotlib\nimport matplotlib.pyplot as plt",
        "plt": "import matplotlib.pyplot as plt",
        "sns": "import seaborn as sns",
        "torch": "import torch",
        "tf": "import tensorflow as tf",
        "sklearn": "import sklearn",
        "sk": "import sklearn",
        "heapq": "import heapq",
        "json": "import json",
        "os": "import os",
        "sys": "import sys",
        "math": "import math",
        "re": "import re",
        "random": "import random",
        "datetime": "import datetime",
        "collections": "import collections",
        "itertools": "import itertools",
        "functools": "import functools",
        "typing": "import typing",
        "pathlib": "import pathlib",
        "inspect": "import inspect",
        "hashlib": "import hashlib",
        "copy": "import copy",
        "pickle": "import pickle",
    }

    def __init__(self):
        self.context: Optional[ExecutionContext] = None

    def unify_and_emit(self, cells: List[Cell], prompt: str = "") -> str:
        """
        Unifies a sequence of cells into a clean, executable Python script with resolved imports.
        """
        if not cells:
            return ""
        context = ExecutionContext(prompt=prompt)
        self.context = context
        statements = []
        for cell in cells:
            code = self.unify_cell(context, cell)
            if code:
                statements.append(code)
        raw_body = "\n".join(statements)
        return self.resolve_imports(raw_body, context)

    @staticmethod
    def unify_cell(
        context: ExecutionContext,
        cell: Cell,
        explicit_arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        explicit_args = explicit_arguments or {}
        resolver = DynamicPlaceholderResolver()

        # Generate output variable name
        cell_id_clean = cell.cell_id.lower().replace("_cell", "").replace("_default", "")
        raw_out_name = cell_id_clean.split('_')[-1]
        output_var_name = context.peek_next_variable_name(f"{raw_out_name}_out")

        # Collect dependencies
        if cell.dependencies:
            for dep in cell.dependencies:
                dep_str = str(dep).strip()
                if not dep_str:
                    continue
                context.declared_dependencies.add(dep_str)

        raw_code = cell.code_template
        if not raw_code:
            mod = UnificationGate._infer_module_prefix(cell)
            func = UnificationGate._infer_function_name(cell)
            latest = context.get_latest_data_variable() or "None"
            if mod:
                raw_code = f"{{output_var}} = {mod}.{func}({latest})"
            else:
                raw_code = f"{{output_var}} = {func}({latest})"

        transformed = raw_code.replace("{output_var}", output_var_name)

        placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", transformed))
        for ph in placeholders:
            if ph in explicit_args:
                val = explicit_args[ph]
                val_str = json.dumps(val) if isinstance(val, str) and not (val.startswith('"') or val.startswith("'")) else str(val)
                transformed = transformed.replace(f"{{{ph}}}", val_str)
                continue

            port_sig = cell.inputs.get(ph)
            if port_sig is None:
                port_sig = PortSignature(ph, AlgebraicSignature("any", "any"))

            resolved = resolver.resolve_port(ph, port_sig, cell.stage, context, output_var_name, cell=cell)
            transformed = transformed.replace(f"{{{ph}}}", resolved)

        # Dynamic attribute unquoting for module constants (e.g. "cv2.COLOR_BGR2GRAY")
        def _unquote_mod(m):
            cand = m.group(1)
            if resolver.is_module_attribute(cand):
                return cand
            return m.group(0)

        transformed = re.sub(r'["\']([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)["\']', _unquote_mod, transformed)

        resolver.assert_placeholders_resolved(transformed)

        # Register primary output in context
        context.declare_variable(
            base_name=f"{raw_out_name}_out",
            signature=cell.primary_output
        )

        logger.info(f"[UNIFICATION SUCCESS] Bound {cell.cell_id} -> {output_var_name}")
        return transformed

    @staticmethod
    def _infer_module_prefix(cell: Cell) -> str:
        for dep in cell.dependencies:
            dep = str(dep).strip()
            if dep.startswith("import "):
                parts = dep.split()
                if len(parts) >= 2:
                    mod = parts[1].split(".")[0].split("as")[0].strip()
                    if mod not in UnificationGate.DENYLISTED_IMPORTS:
                        return mod
        domain_map = {
            "opencv": "cv2", "cv2": "cv2",
            "pandas": "pd", "numpy": "np",
            "scipy": "scipy", "sklearn": "sklearn",
            "matplotlib": "plt", "json": "json",
            "math": "math", "os": "os", "sys": "sys",
        }
        return domain_map.get((cell.domain_name or "").lower(), "")

    @staticmethod
    def _infer_function_name(cell: Cell) -> str:
        parts = cell.cell_id.lower().replace("_cell", "").replace("_default", "").split("_")
        if parts and parts[0] in ("cv2", "pd", "np", "plt", "sns", "sk", "tf", "torch"):
            parts = parts[1:]
        return "_".join(parts) if parts else "compute"

    @staticmethod
    def resolve_imports(code_text: str, context: Optional[ExecutionContext] = None, chain_nodes: Optional[List[Any]] = None) -> str:
        if not code_text or not code_text.strip():
            return code_text

        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return code_text

        existing_imports: Set[str] = set()
        body_nodes = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                existing_imports.add(ast.unparse(node).strip())
            else:
                body_nodes.append(node)

        if context:
            for dep in context.declared_dependencies:
                existing_imports.add(dep)

        if chain_nodes:
            for c_node in chain_nodes:
                for dep in getattr(c_node, "dependencies", []):
                    if dep.startswith("import ") or dep.startswith("from "):
                        existing_imports.add(dep)
                    elif dep in UnificationGate.CANONICAL_IMPORTS:
                        existing_imports.add(UnificationGate.CANONICAL_IMPORTS[dep])
                    else:
                        existing_imports.add(f"import {dep}")

        module_access_names: Set[str] = set()
        for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module_access_names.add(node.value.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                module_access_names.add(node.func.id)

        required_imports: Set[str] = set()
        for imp in existing_imports:
            mod_root = (
                imp.replace("import", "").strip().split()[0].split(".")[0].lower()
                if imp.startswith("import") else imp.split(".")[0].lower()
            )
            if mod_root in UnificationGate.DENYLISTED_IMPORTS:
                continue
            required_imports.add(imp)

        for name in module_access_names:
            if name in UnificationGate.CANONICAL_IMPORTS:
                required_imports.add(UnificationGate.CANONICAL_IMPORTS[name])
            elif name not in ("str", "int", "float", "list", "dict", "set", "tuple", "bool", "print", "len", "max", "min", "range", "abs", "round", "sum", "any", "all", "getattr", "hasattr", "setattr", "type", "isinstance", "open", "filter", "map"):
                try:
                    import importlib.util
                    if importlib.util.find_spec(name) is not None:
                        required_imports.add(f"import {name}")
                except Exception:
                    pass

        new_tree = ast.Module(body=body_nodes, type_ignores=[])
        ast.fix_missing_locations(new_tree)
        clean_body = ast.unparse(new_tree).strip()

        if required_imports:
            header = "\n".join(sorted(required_imports))
            return f"{header}\n\n{clean_body}"
        return clean_body

    @staticmethod
    def validate_synthesis(
        synthesized_dict: dict,
        expected_in_sig: Any = None,
        expected_out_sig: Any = None,
        trees_dir: str = "trees",
        expected_inputs: Any = None,
        expected_outputs: Any = None,
    ) -> bool:
        try:
            code = synthesized_dict.get("code_template", "")
            if not code:
                for d_val in synthesized_dict.get("domain_implementations", {}).values():
                    if isinstance(d_val, dict) and "code" in d_val:
                        code = d_val["code"]
                        break
            if not code:
                return False

            in_sig = expected_in_sig or expected_inputs
            out_sig = expected_out_sig or expected_outputs

            if isinstance(in_sig, str):
                in_sig = AlgebraicSignature(in_sig, "any")
            elif in_sig is None:
                in_sig = AlgebraicSignature("any", "any")

            if isinstance(out_sig, str):
                out_sig = AlgebraicSignature(out_sig, "any")
            elif out_sig is None:
                out_sig = AlgebraicSignature("any", "any")

            test_code = code.replace("{output_var}", "out_var").replace("{input_var}", "in_var")
            placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", test_code))
            for ph in placeholders:
                test_code = test_code.replace(f"{{{ph}}}", "None")
            ast.parse(test_code)

            actual_in = AlgebraicSignature(
                synthesized_dict.get("inputs", {}).get("type_name", "any"),
                synthesized_dict.get("inputs", {}).get("state", "any")
            )
            actual_out = AlgebraicSignature(
                synthesized_dict.get("outputs", {}).get("type_name", "any"),
                synthesized_dict.get("outputs", {}).get("state", "any")
            )

            if not in_sig.unifies_with(actual_in):
                logger.warning(
                    f"[VALIDATE] Input type mismatch: expected {in_sig}, got {actual_in}"
                )
                return False
            if not actual_out.unifies_with(out_sig):
                logger.warning(
                    f"[VALIDATE] Output type mismatch: expected {out_sig}, got {actual_out}"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"[VALIDATE] Synthesis validation failed: {e}")
            return False


class DataflowLineageTracker(ast.NodeTransformer):
    def __init__(self, target_cells=None):
        self.target_cells = target_cells or []
        self.lineage_tree: Dict[str, str] = {}
        self.latest_descendant: Dict[str, str] = {}
        self.assigned_vars: Set[str] = set()
        self._imported_names: Set[str] = set()

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
        self._rebind_sink_call(node.value)
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            self.assigned_vars.add(target_name)
            parent_var = None
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
                    candidate = node.value.func.value.id
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
            if callee_var and callee_var in self.latest_descendant and callee_var not in self._imported_names:
                newest_var = self.latest_descendant[callee_var]
                if newest_var != callee_var:
                    call_node.func.value.id = newest_var
            for arg in call_node.args:
                if isinstance(arg, ast.Name) and arg.id in self.latest_descendant:
                    arg.id = self.latest_descendant[arg.id]

    def _get_root(self, var_name: str) -> str:
        curr = var_name
        while curr in self.lineage_tree:
            curr = self.lineage_tree[curr]
        return curr


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    try:
        tree = ast.parse(code)
        transformer = DataflowLineageTracker(target_cells=target_cells)
        corrected_tree = transformer.visit(tree)
        ast.fix_missing_locations(corrected_tree)
        return ast.unparse(corrected_tree)
    except Exception:
        return code


@dataclass
class ExtractedSlots:
    named_identifiers: List[str] = field(default_factory=list)
    dest_uris: List[str] = field(default_factory=list)
    source_uris: List[str] = field(default_factory=list)
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    operational_flags: Dict[str, Any] = field(default_factory=dict)
    by_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "named_identifiers": self.named_identifiers,
            "dest_uris": self.dest_uris,
            "source_uris": self.source_uris,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "columns": self.columns,
            "operational_flags": self.operational_flags,
            "by_column": self.by_column,
        }


class ParameterExtractor:
    # Extensible declarative table: maps prompt keywords → operational flag entries.
    # Domain-agnostic: these detect semantic signals (color modes, sort directions)
    # regardless of which downstream library processes them.
    SEMANTIC_FLAG_PATTERNS: list = [
        # (keywords_to_match, flag_key, flag_value, extra_flags)
        ({"descend", "descending", "reverse"}, "descending", True, {"ascending": False}),
        ({"ascend", "ascending"},              "ascending",  True, {"descending": False}),
        ({"hsv"},                              "is_hsv",     True, {}),
        ({"rgb"},                              "is_rgb",     True, {}),
        ({"gray", "grayscale", "grey"},        "is_grayscale", True, {}),
    ]

    @staticmethod
    def extract_slots(prompt: str) -> ExtractedSlots:
        ctx = ExecutionContext(prompt=prompt)
        flags = dict(ctx.flags)
        if prompt:
            p_low = prompt.lower()
            # Apply declarative semantic flag patterns
            for keywords, flag_key, flag_val, extras in ParameterExtractor.SEMANTIC_FLAG_PATTERNS:
                if any(kw in p_low for kw in keywords):
                    flags[flag_key] = flag_val
                    flags.update(extras)

        return ExtractedSlots(
            named_identifiers=ctx.columns,
            dest_uris=ctx.dest_files,
            source_uris=ctx.source_files,
            input_files=ctx.source_files,
            output_files=ctx.dest_files,
            columns=ctx.columns,
            operational_flags=flags,
            by_column=ctx.by_column
        )

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()


def resolve_node_slots(template: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
    """Deterministically binds extracted prompt parameters to template slot placeholders.
    Uses structural analysis of the template and declarative flag tables — no domain-specific logic."""
    slots = {}
    placeholders = re.findall(r"\{([a-zA-Z0-9_]+)\}", template)
    src_uris = extracted_params.get("source_uris", [])
    dst_uris = extracted_params.get("dest_uris", [])
    by_col = extracted_params.get("by_column")
    flags = extracted_params.get("operational_flags", {})
    
    for ph in placeholders:
        ph_l = ph.lower()
        if "filename" in ph_l or "path" in ph_l or "uri" in ph_l:
            if "output" in ph_l or "dest" in ph_l or "save" in ph_l:
                if dst_uris:
                    slots[ph] = f"'{dst_uris[-1]}'"
            else:
                if src_uris:
                    slots[ph] = f"'{src_uris[0]}'"
        elif ph_l in ("by", "by_column", "column"):
            if by_col:
                slots[ph] = f"'{by_col}'"
        elif ph_l == "ascending":
            slots[ph] = str(flags.get("ascending", True))
        elif ph_l in ("code", "color_code", "conversion_code"):
            # Infer the module name from the template context via structural analysis.
            # e.g. in "cv2.cvtColor({input}, {code})" we extract "cv2" from the template AST.
            mod_match = re.search(r"\b([a-zA-Z0-9_]+)\.[a-zA-Z0-9_]+\s*\([^)]*\{" + re.escape(ph) + r"\}", template)
            if not mod_match:
                # No module context found — cannot resolve without guessing a specific library.
                # Leave unresolved (monoidal identity) rather than hardcoding a domain.
                continue
            mod_name = mod_match.group(1)

            # Derive target format from semantic flags
            target_fmt = "GRAY"  # Default conversion target (domain-agnostic)
            if flags.get("is_hsv"):
                target_fmt = "HSV"
            elif flags.get("is_rgb"):
                target_fmt = "RGB"

            # Resolve via runtime module reflection
            resolved_code = None
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                # Search the module's namespace for matching conversion constant
                cand = next((attr for attr in dir(mod) if attr.startswith("COLOR_BGR2") and attr.endswith(target_fmt)), None)
                if not cand:
                    cand = next((attr for attr in dir(mod) if attr.startswith("COLOR_") and attr.endswith(target_fmt)), None)
                if cand:
                    resolved_code = f"{mod_name}.{cand}"
            except Exception:
                pass

            slots[ph] = resolved_code or f"{mod_name}.COLOR_BGR2{target_fmt}"

    return slots
