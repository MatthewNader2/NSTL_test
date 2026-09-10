"""
src/synthesis.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamic MicroCell Synthesizer: Grounded in Live API Documentation.
"""

from __future__ import annotations
import ast
import json
import os
from typing import Dict, Any, List, Optional
from log_config import get_logger

try:
    from .external_rag import LiveDocFetcher
    from .inference import ModelManager
    from .utils import extract_json_from_llm, validate_code_template, extract_code_from_llm_response
except (ImportError, ValueError):
    from external_rag import LiveDocFetcher
    from inference import ModelManager
    from utils import extract_json_from_llm, validate_code_template, extract_code_from_llm_response

logger = get_logger('synthesis')


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
        stage: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Queries official API documentation and uses the LLM to synthesize
        a verified, type-annotated MicroCell.
        """
        logger.info(f"[SYNTHESIS] Fetching live documentation for: '{gap_concept}'")
        live_docs = fetcher.fetch(gap_concept) or "No live documentation available."

        # Infer stage algebraically from input/output spec if not explicitly passed
        if stage is None:
            if not expected_input or str(expected_input).lower() in ("any", "none"):
                stage = 1
            elif not expected_output or str(expected_output).lower() in ("none", "void", "noreturn"):
                stage = 3
            else:
                stage = 2

        if stage == 1:
            template_rule = "For Stage 1 (file loading/source), `code_template` MUST use `{filepath}` as the input path argument and `{output_var}` for the output assignment (e.g. `{output_var} = package.load_func({filepath})`)."
            input_spec = '{"type_name": "str", "state": "source_identifier"}'
            sample_template = "{output_var} = <package>.<source_func>({filepath})"
        elif stage == 3:
            template_rule = "For Stage 3 (exporting/saving), `code_template` MUST use `{dest_path}` as the destination file path (e.g. `{input_var}.save_func({dest_path})\\n{output_var} = {dest_path}`)."
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{input_var}.<sink_func>({dest_path})\\n{output_var} = {dest_path}"
        else:
            template_rule = "For Stage 2 (data transform), `code_template` MUST use `{input_var}` for input and `{output_var}` for output (e.g. `{output_var} = <func>({input_var})`)."
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{output_var} = <package>.<func>({input_var})"

        sanitized_concept = "".join(c if c.isalnum() or c == "_" else "_" for c in gap_concept).lower()[:30]
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
            logger.error(f"[SYNTHESIS ERROR] Failed to extract JSON from model output")
            raise ValueError(f"Model failed to generate valid JSON for {gap_concept}")

        # Validate template syntax by substituting dummy identifiers for all placeholders
        template = extract_code_from_llm_response(cell_dict.get("code_template", ""))
        cell_dict["code_template"] = template
        if not validate_code_template(template):
            logger.error(f"[SYNTHESIS ERROR] Synthesized template failed AST validation")
            raise ValueError(f"Synthesized template for {gap_concept} failed AST parse check.")

        return cell_dict


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
        raw_t = str(getattr(getattr(p_sig, "signature", None), "type_name", "") or getattr(p_sig, "type_name", ""))
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


def render_cell(
    cell: Any,
    bindings: Dict[str, Any],
    indent_level: int = 0,
    context: Optional[Any] = None,
    accumulated_sigma: Optional[Any] = None
) -> str:
    """
    Renders an executable block of Python code for a cell at the specified indentation level.
    Recursively renders child sub-pipelines in bound_slots at indent_level + 1.
    Handles invariant accumulator wiring for traced loops and coproduct branch joins.

    All loop semantics are driven by DECLARED data only:
      - The loop item carrier is recovered from the container's generic argument.
      - The extremum direction comes from the cell's declared direction port binding
        (resolved through the typestate polarity machinery), never from prompt text.
      - Collection loops are detected structurally from the declared template's
        container initialization, never from cell identifier strings.
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

    out_var = bindings.get("output_var", f"_{getattr(cell.primary_output, 'state', None) or 'out'}")

    # Structural discriminator: the declared template initializes a collect container.
    is_collect_loop = "{output_var} = []" in template

    # Declared extremum direction: bound value of the cell's declared direction port.
    # The port's polarity vocabulary lives in the tree schema (qualifiers); the value
    # here was resolved by the typestate polarity projection in ExecutionContext.
    direction_raw = bindings.get("direction")
    is_extremum_loop = "direction" in getattr(cell, "inputs", {}) and direction_raw is not None
    is_max = str(direction_raw).strip().lower().lstrip("'").rstrip("'") == "true" if direction_raw is not None else False

    # Declared iteration carrier for child port wiring (monoidal container recovery).
    item_type = _derive_loop_item_type(cell, accumulated_sigma) if (is_collect_loop or is_extremum_loop) else None

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
        last_metric = "_res"
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
                            if unify(item_type, child_sig, Substitution()) is not None:
                                child_bindings[p_name] = "_item"
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

            child_out_var = f"_{getattr(child.primary_output, 'state', None) or 'res'}"
            child_bindings["output_var"] = child_out_var
            last_metric = child_out_var

            child_text = UnificationGate._instantiate_ast_template(child.code_template, child_bindings, child.inputs)
            if child_text:
                child_lines.append(child_text)

        # Traced loop reduction logic: if the macro declares an extremum direction
        # port, emit the invariant accumulator update for the declared direction.
        topology = getattr(cell, "topology_type", "sequential")
        if is_extremum_loop:
            op = ">" if is_max else "<"
            metric_var = "_max_metric" if is_max else "_min_metric"

            update_block = (
                f"if {metric_var} is None or {last_metric} {op} {metric_var}:\n"
                f"    {metric_var} = {last_metric}\n"
                f"    {out_var} = _item"
            )
            child_lines.append(update_block)

        elif is_collect_loop:
            child_lines.append(f"{out_var}.append({last_metric})")

        slot_code = "\n".join(child_lines)
        slot_rendered[slot_name] = slot_code

    rendered = template
    if is_extremum_loop:
        metric_var = "_max_metric" if is_max else "_min_metric"
        if f"{metric_var} = None" not in rendered:
            rendered = f"{metric_var} = None\n" + rendered

    for slot_name, slot_code in slot_rendered.items():
        placeholder = f"{{{slot_name}}}"
        import re
        match = re.search(rf"^([ \t]*)\{{{slot_name}\}}", rendered, flags=re.MULTILINE)
        if match:
            base_indent = match.group(1) or "    "
            indented_slot = "\n".join(
                (base_indent + line if line.strip() else line)
                for line in slot_code.splitlines()
            )
            rendered = rendered.replace(match.group(0), indented_slot)
        elif placeholder in rendered:
            indented_slot = "\n".join(
                ("    " + line if line.strip() else line)
                for line in slot_code.splitlines()
            )
            rendered = rendered.replace(placeholder, indented_slot)

    # Instantiate remaining placeholders using bindings
    for k, v in bindings.items():
        if v is not None:
            rendered = rendered.replace(f"{{{k}}}", str(v))

    # Apply base indent_level
    if indent_level > 0:
        rendered = "\n".join(
            (indent_prefix + line if line.strip() else line)
            for line in rendered.splitlines()
        )

    return rendered


def build_script(
    cells: List[Any],
    context: Optional[Any] = None
) -> str:
    """
    Emits a fully verified, block-structured Python script from a cell pipeline.
    """
    try:
        from .unification import UnificationGate, ExecutionContext, Success, Failure
    except (ImportError, ValueError):
        from unification import UnificationGate, ExecutionContext, Success, Failure

    gate = UnificationGate()
    ctx = context or ExecutionContext()
    res = gate.unify_pipeline(cells, ctx)
    if res.is_bottom():
        reason = res.reason if isinstance(res, Failure) else "Unification bottom"
        raise ValueError(f"Unification Failed: {reason}")

    assert isinstance(res, Success)
    pipeline_bindings = res.value

    # Collect dependencies recursively
    deps: List[str] = []
    def collect_deps(c: Any):
        for dep in getattr(c, "dependencies", []):
            dep_clean = dep.strip()
            if dep_clean and dep_clean not in deps:
                deps.append(dep_clean)
        for sub_list in getattr(c, "bound_slots", {}).values():
            if isinstance(sub_list, list):
                for sc in sub_list:
                    collect_deps(sc)

    for cell, _ in pipeline_bindings:
        collect_deps(cell)

    code_lines: List[str] = []
    if deps:
        code_lines.extend(deps)
        code_lines.append("")

    for cell, bindings in pipeline_bindings:
        rendered = render_cell(cell, bindings, indent_level=0, context=ctx, accumulated_sigma=res.sigma)
        if rendered:
            code_lines.append(rendered)

    final_code = "\n".join(code_lines).strip()
    return final_code
