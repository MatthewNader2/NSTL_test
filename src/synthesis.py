"""
src/synthesis.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamic MicroCell Synthesizer: Grounded in Live API Documentation.
"""

from __future__ import annotations
import asyncio
import ast
import importlib
import json
import os
import textwrap
from typing import Dict, Any, List, Optional, Tuple
from log_config import get_logger

try:
    from .external_rag import LiveDocFetcher
    from .inference import ModelManager
    from .utils import (
        extract_json_from_llm,
        validate_code_template,
        extract_code_from_llm_response,
        safe_substitute_template,
        extract_template_placeholders,
    )
except (ImportError, ValueError):
    from external_rag import LiveDocFetcher
    from inference import ModelManager
    from utils import (
        extract_json_from_llm,
        validate_code_template,
        extract_code_from_llm_response,
        safe_substitute_template,
        extract_template_placeholders,
    )

logger = get_logger('synthesis')


# --------------------------------------------------------------------------- #
# Combinator-library detection
# --------------------------------------------------------------------------- #

def import_stmt_for(qualified_name: str) -> str:
    """
    Derives an import statement from a qualified symbol or module name, verified live.
    Conforms to Section 3.4 of the NSTL paper.
    """
    module = qualified_name.rsplit(".", 1)[0] if "." in qualified_name else qualified_name
    try:
        importlib.import_module(module)
    except ImportError:
        logger.warning("unverifiable import: %s", module)
    return f"import {module}"


def _safe_substitute(template: str, bindings: Dict[str, Any]) -> str:
    """
    Placeholder substitution that ONLY touches ``{identifier}`` spans.
    Uses zero-regex safe_substitute_template.
    """
    return safe_substitute_template(template, bindings)


# --------------------------------------------------------------------------- #
# Micro-cell synthesis
# --------------------------------------------------------------------------- #

class SynthesisEngine:
    """
    Synthesizes missing computational primitives on-demand using live documentation lookup.
    """

    def __init__(self, trees_dir: str = "trees"):
        self.trees_dir = trees_dir

    def synthesize_micro_cell(
        self,
        gap_concept: str,
        expected_input: str,
        expected_output: str,
        fetcher: LiveDocFetcher,
        domain: str = "Python_Core",
        stage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Queries official API documentation and uses the LLM to synthesize
        a verified, type-annotated MicroCell.
        """
        logger.info(f"[SYNTHESIS] Fetching live documentation for: '{gap_concept}'")
        # LiveDocFetcher exposes a synchronous text interface; the async
        # registry path is resolved internally (thread-offloaded when a loop
        # is already running). Result is plain documentation text.
        fetch_call = getattr(fetcher, "fetch", None)
        if fetch_call is None:
            live_docs = ""
        else:
            try:
                live_docs = fetch_call(gap_concept)
            except TypeError:
                live_docs = ""
            # Guard against awaitable returns from async fetchers.
            if asyncio.iscoroutine(live_docs) or isinstance(live_docs, asyncio.Future):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    live_docs = asyncio.run_coroutine_threadsafe(live_docs, loop).result()
                else:
                    live_docs = asyncio.run(live_docs)
            if not isinstance(live_docs, str):
                live_docs = str(getattr(live_docs, "content", "") or "")
        live_docs = live_docs or "No live documentation available."

        # Infer stage algebraically from input/output spec if not explicitly passed
        if stage is None:
            if not expected_input or str(expected_input).lower() in ("any", "none"):
                stage = 1
            elif not expected_output or str(expected_output).lower() in ("none", "void", "noreturn"):
                stage = 3
            else:
                stage = 2

        if stage == 1:
            template_rule = (
                "For Stage 1 (file loading/source), `code_template` MUST use `{filepath}` "
                "as the input path argument and `{output_var}` for the output assignment "
                "(e.g. `{output_var} = package.load_func({filepath})`)."
            )
            input_spec = '{"type_name": "str", "state": "source_identifier"}'
            sample_template = "{output_var} = <package>.<source_func>({filepath})"
        elif stage == 3:
            template_rule = (
                "For Stage 3 (exporting/saving), `code_template` MUST use `{dest_path}` as "
                "the destination file path (e.g. `{input_var}.save_func({dest_path})\\n"
                "{output_var} = {dest_path}`)."
            )
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = (
                "{input_var}.<sink_func>({dest_path})\n{output_var} = {dest_path}"
            )
        else:
            template_rule = (
                "For Stage 2 (data transform), `code_template` MUST use `{input_var}` for "
                "input and `{output_var}` for output (e.g. `{output_var} = <func>({input_var})`)."
            )
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{output_var} = <package>.<func>({input_var})"

        sanitized_concept = "".join(
            c if c.isalnum() or c == "_" else "_" for c in gap_concept
        ).lower()[:30]

        system_prompt = f"""You are an expert Software Engineer. Synthesize a single verified Python computational node implementing: '{gap_concept}'.
Output ONLY a valid JSON object matching the schema below. No markdown formatting, no explanations.

Schema:
{{
  "cell_id": "micro_synthesized_{sanitized_concept}",
  "type": "micro",
  "stage": {stage},
  "keywords": ["{gap_concept}"],
  "inputs": {input_spec},
  "outputs": {{ "type_name": "{expected_output}", "state": "computed" }},
  "dependencies": ["<import statement>"],
  "code_template": "{sample_template}"
}}

Documentation Reference:
{live_docs[:1500]}

RULES:
1. {template_rule}
2. Put all necessary import statements in the `dependencies` array (e.g. ["import package", "from package import module"]).
3. Ensure the code is strictly non-interactive (do not use input() or sys.stdin).
4. Never hardcode file asset paths — always use `{{filepath}}` or `{{dest_path}}` placeholders."""

        full_prompt = f"{system_prompt}\n\nTask: {gap_concept}"

        result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=1024)

        cell_dict = extract_json_from_llm(result_text)
        if cell_dict is None:
            logger.error("[SYNTHESIS ERROR] Failed to extract JSON from model output")
            raise ValueError(f"Model failed to generate valid JSON for {gap_concept}")

        # Validate template syntax by substituting dummy identifiers for all placeholders
        template = extract_code_from_llm_response(cell_dict.get("code_template", ""))
        cell_dict["code_template"] = template
        if not validate_code_template(template):
            logger.error("[SYNTHESIS ERROR] Synthesized template failed AST validation")
            raise ValueError(f"Synthesized template for {gap_concept} failed AST parse check.")

        return cell_dict


def synthesize_and_register(
    engine: SynthesisEngine,
    gap_concept: str,
    expected_input: str,
    expected_output: str,
    fetcher: LiveDocFetcher,
    registry: Any,
    domain: str = "Python_Core",
    stage: Optional[int] = None,
) -> Optional[Any]:
    """
    Pipeline-facing entrypoint that synthesizes a micro-cell and registers it
    into a TypeRegistry / cell repository. Wires SynthesisEngine into the
    active pipeline without forcing callers to know the internal shape.
    """
    try:
        cell_dict = engine.synthesize_micro_cell(
            gap_concept=gap_concept,
            expected_input=expected_input,
            expected_output=expected_output,
            fetcher=fetcher,
            domain=domain,
            stage=stage,
        )
    except Exception as exc:
        logger.error("[SYNTHESIS] registration failed for '%s': %s", gap_concept, exc)
        return None

    if registry is None:
        return cell_dict

    for method_name in ("register_cell", "add_cell", "register"):
        method = getattr(registry, method_name, None)
        if callable(method):
            try:
                method(cell_dict)
                logger.info("[SYNTHESIS] registered synthesized cell '%s'", cell_dict.get("cell_id"))
                return cell_dict
            except Exception as exc:
                logger.error("[SYNTHESIS] registry.%s failed: %s", method_name, exc)
                return cell_dict

    logger.warning("[SYNTHESIS] no compatible register method on registry; returning raw dict")
    return cell_dict


# --------------------------------------------------------------------------- #
# Loop rendering helpers (all data-driven, zero hard-coded variable names)
# --------------------------------------------------------------------------- #

def _derive_loop_item_type(cell: Any, accumulated_sigma: Optional[Any]) -> Optional[Any]:
    """
    Derives the declared iteration carrier (item type) of a traced-loop macro.
    Categorically: a loop morphism consumes a monoidal container C[T]; the item
    carrier T is recovered from the generic argument of the container-typed input
    port after generic substitution. Works for ANY container C and item type T
    with zero hardcoded container or item names. Returns None if not derivable.
    """
    try:
        from .unification import TypeTerm, GenericTypeTerm, substitute_generics
    except (ImportError, ValueError):
        from unification import TypeTerm, GenericTypeTerm, substitute_generics

    for p_sig in getattr(cell, "inputs", {}).values():
        raw_t = str(
            getattr(getattr(p_sig, "signature", None), "type_name", "")
            or getattr(p_sig, "type_name", "")
        )
        if not raw_t or "[" not in raw_t:
            continue
        concrete_t = substitute_generics(raw_t, accumulated_sigma) if accumulated_sigma is not None else raw_t
        try:
            term = TypeTerm.from_string(concrete_t)
        except Exception:
            continue
        if isinstance(term, GenericTypeTerm) and term.args:
            return term.args[0]
    return None


def _declared_loop_kind(cell: Any) -> Optional[str]:
    """
    Reads the declared loop kind from cell metadata (never from template strings).
    Returns 'collect', 'extremum', or None.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "loop_kind", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("loop_kind")
    if declared in ("collect", "extremum"):
        return declared

    topology = getattr(cell, "topology_type", None)
    if topology in ("collect_loop", "collection_loop", "accumulate_loop"):
        return "collect"
    if topology in ("extremum_loop", "reduce_loop", "argmax_loop", "argmin_loop"):
        return "extremum"
    return None


def _template_initializes_container(template: str) -> bool:
    """
    Structural fallback: detect whether the template initializes the output
    container variable to an empty container. Accepts any literal container
    expression or empty-constructor call, without matching one hardcoded
    literal (previously `{output_var} = []`).
    """
    marker = "{output_var}"
    constructors = (
        "set", "dict", "list", "tuple", "frozenset",
        "deque", "OrderedDict", "defaultdict", "Counter",
    )
    for line in template.splitlines():
        s = line.strip()
        idx = s.find(marker)
        if idx < 0:
            continue
        tail = s[idx + len(marker):].lstrip()
        if not tail.startswith("="):
            continue
        rhs = tail[1:].strip()
        if not rhs:
            continue
        if rhs[0] in "([{":
            return True
        if rhs.endswith("()") and rhs[:-2].strip() in constructors:
            return True
    return False


def _detect_loop_item_var(template: str) -> Optional[str]:
    """
    Scans the template for a `for <var> in ...` header and returns the loop
    variable name. Handles tuple-unpacking targets by taking the first name.
    """
    for line in template.splitlines():
        s = line.strip()
        if not s.startswith("for ") or " in " not in s:
            continue
        head = s[4:].split(" in ", 1)[0].strip()
        if "," in head:
            head = head.split(",", 1)[0].strip()
        head = head.strip("()").strip()
        if head.isidentifier():
            return head
    return None


def _unique_var(cell: Any, suffix: str) -> str:
    """
    Returns a cell-scoped variable name of the form `_<cell_id>_<suffix>`.
    Guarantees no collision between nested renders of distinct cells.
    """
    cid = (
        getattr(cell, "cell_id", None)
        or getattr(cell, "name", None)
        or "cell"
    )
    safe = "".join(c if c.isalnum() else "_" for c in str(cid)) or "cell"
    return f"_{safe}_{suffix}"


def _resolve_item_var(cell: Any, template: str) -> str:
    """
    Resolves the loop item variable name from declared metadata, or detected
    from the template's `for ... in ...` header, or a cell-scoped fallback.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "item_var", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("item_var")
    if declared:
        return str(declared)
    detected = _detect_loop_item_var(template)
    if detected:
        return detected
    return _unique_var(cell, "item")


def _resolve_direction_is_max(cell: Any, bindings: Dict[str, Any]) -> bool:
    """
    Resolves the declared extremum direction from a bound port value. Only
    reads declared bindings (never prompt text). Returns True for max/argmax.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    direction_port = (
        getattr(cell, "direction_port", None)
        or (meta.get("direction_port") if isinstance(meta, dict) else None)
        or "direction"
    )
    raw = bindings.get(direction_port)
    if raw is None:
        for k in ("direction", "extreme", "mode", "order"):
            if k in bindings and bindings[k] is not None:
                raw = bindings[k]
                break
    if raw is None:
        return True
    s = str(raw).strip().strip("'\"").lower()
    if s in ("true", "max", "maximum", "argmax", "gt", ">", "1", "desc", "descending"):
        return True
    if s in ("false", "min", "minimum", "argmin", "lt", "<", "0", "asc", "ascending"):
        return False
    return True


def render_cell(
    cell: Any,
    bindings: Dict[str, Any],
    indent_level: int = 0,
    context: Optional[Any] = None,
    accumulated_sigma: Optional[Any] = None,
) -> str:
    """
    Renders an executable block of Python code for a cell at the specified indentation level.
    Recursively renders child sub-pipelines in bound_slots at indent_level + 1.
    Handles invariant accumulator wiring for traced loops and coproduct branch joins.

    All loop semantics are driven by DECLARED data only:
      - The loop kind (collect vs extremum) comes from cell metadata
        (`loop_kind` / `topology_type`); a structural container-init scan is used
        only as a last-resort fallback.
      - The loop item carrier is recovered from the container's generic argument.
      - The extremum direction comes from the cell's declared direction port
        binding (resolved through the typestate polarity machinery), never from
        prompt text.
      - All injected helper variables are cell-scoped unique names derived from
        the cell identifier; no hardcoded globals like `_item`, `_res`,
        `_max_metric`, `_min_metric`.
    """
    try:
        from .unification import UnificationGate, unify, Substitution
    except (ImportError, ValueError):
        from unification import UnificationGate, unify, Substitution

    indent_prefix = "    " * indent_level
    template = cell.code_template.strip()
    if not template:
        return ""

    slots = getattr(cell, "slots", {}) or {}
    bound_slots = getattr(cell, "bound_slots", {}) or {}

    out_var = bindings.get(
        "output_var",
        f"_{getattr(cell.primary_output, 'state', None) or 'out'}",
    )

    # ---- Declared loop semantics ----
    loop_kind = _declared_loop_kind(cell)
    if loop_kind == "extremum":
        is_extremum_loop, is_collect_loop = True, False
    elif loop_kind == "collect":
        is_extremum_loop, is_collect_loop = False, True
    else:
        # Structural fallback, driven by declared ports, not by prompt text.
        has_direction_port = "direction" in getattr(cell, "inputs", {})
        direction_bound = bindings.get("direction") is not None
        is_extremum_loop = has_direction_port and direction_bound
        is_collect_loop = (
            not is_extremum_loop and _template_initializes_container(template)
        )

    is_max = _resolve_direction_is_max(cell, bindings) if is_extremum_loop else False

    # ---- Scoped, unique variable names (no hardcoded globals) ----
    item_var = _resolve_item_var(cell, template)
    metric_var = _unique_var(cell, "max_metric" if is_max else "min_metric")

    # Declared iteration carrier for child port wiring (monoidal container recovery).
    item_type = (
        _derive_loop_item_type(cell, accumulated_sigma)
        if (is_collect_loop or is_extremum_loop)
        else None
    )

    # Process each declared slot
    slot_rendered: Dict[str, str] = {}
    for slot_name in slots.keys():
        child_nodes = bound_slots.get(slot_name, [])
        if not child_nodes:
            slot_rendered[slot_name] = "pass"
            continue

        if isinstance(child_nodes, str):
            slot_rendered[slot_name] = child_nodes
            continue

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        child_lines: List[str] = []
        last_metric: Optional[str] = None
        for child in child_nodes:
            child_bindings: Dict[str, Any] = {}
            for p_name, p_sig in child.inputs.items():
                # 1. Type-driven loop-item wiring: bind the child's port to the loop
                #    variable iff the port's declared signature unifies with the item
                #    carrier recovered from the container's generic argument.
                bound = False
                if item_type is not None:
                    child_sig = getattr(p_sig, "signature", None)
                    if child_sig is not None:
                        try:
                            if (
                                unify(item_type, child_sig, Substitution()) is not None
                                or unify(child_sig, item_type, Substitution()) is not None
                            ):
                                child_bindings[p_name] = item_var
                                bound = True
                        except Exception:
                            bound = False
                if bound:
                    continue
                # 2. Ports matching a parent binding keep the parent's wire.
                if p_name in bindings:
                    child_bindings[p_name] = bindings[p_name]
                elif getattr(p_sig, "default_value", None) is not None:
                    child_bindings[p_name] = str(p_sig.default_value)
                else:
                    child_bindings[p_name] = p_name

            child_out_var = _unique_var(child, "out")
            child_bindings["output_var"] = child_out_var
            last_metric = child_out_var

            child_text = UnificationGate._instantiate_ast_template(
                child.code_template, child_bindings, child.inputs
            )
            if child_text:
                child_lines.append(child_text)

        # Traced loop reduction logic, driven purely by declared loop kind.
        if is_extremum_loop and last_metric is not None:
            op = ">" if is_max else "<"
            update_block = (
                f"if {metric_var} is None or {last_metric} {op} {metric_var}:\n"
                f"    {metric_var} = {last_metric}\n"
                f"    {out_var} = {item_var}"
            )
            child_lines.append(update_block)
        elif is_collect_loop and last_metric is not None:
            child_lines.append(f"{out_var}.append({last_metric})")

        slot_rendered[slot_name] = "\n".join(child_lines) if child_lines else "pass"

    rendered = template
    if is_extremum_loop and f"{metric_var} = None" not in rendered:
        rendered = f"{metric_var} = None\n" + rendered

    for slot_name, slot_code in slot_rendered.items():
        placeholder = f"{{{slot_name}}}"
        if placeholder not in rendered:
            continue
        base_indent = "    "
        found_target = None
        for line in rendered.splitlines():
            stripped = line.lstrip(" \t")
            if stripped.startswith(placeholder):
                base_indent = line[: len(line) - len(stripped)]
                found_target = base_indent + placeholder
                break

        indented_slot = "\n".join(
            (base_indent + line if line.strip() else line)
            for line in slot_code.splitlines()
        )
        if found_target and found_target in rendered:
            rendered = rendered.replace(found_target, indented_slot)
        else:
            rendered = rendered.replace(placeholder, indented_slot)

    # Instantiate remaining placeholders using bindings safely
    rendered = safe_substitute_template(rendered, bindings)

    # Apply base indent_level
    if indent_level > 0:
        rendered = "\n".join(
            (indent_prefix + line if line.strip() else line)
            for line in rendered.splitlines()
        )

    return rendered


__all__ = [
    "SynthesisEngine",
    "synthesize_and_register",
    "render_cell",
    "import_stmt_for",
]
