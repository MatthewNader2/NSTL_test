"""
src/semantic_repair_engine.py

General dynamic semantic validation and repair engine for NSTL cells.
Contains ZERO hardcoded library checks or cell-name checks.

Given ANY cell in ANY tree:
1. Dynamically resolves the runtime callable invoked in `code_template`.
2. Validates the call against the ground truth signature contract (inspect / docstrings).
3. Discards spurious enums that do not belong to the callable.
4. Restores missing or clobbered required positional parameters.
5. Injects valid enums into their declared parameter slots (positional or keyword).
6. Synchronizes `inputs` with template placeholders.
7. Clears LLM docstring fields if semantic changes occurred on an LLM-enriched cell.
"""

from typing import Any, Dict, List, Optional
from src.signature_introspector import (
    parse_call_expression,
    resolve_callable_from_expr,
    validate_and_reconstruct_call,
)
from src.template_wiring import (
    clean_malformed_template_braces,
    repair_wiring_invariant,
)


def repair_cell_semantics(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Dynamically validates and repairs a single cell against ground-truth runtime signatures.

    Returns True if the cell was modified.
    """
    modified = False

    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    if tmpl != cell.get("code_template", ""):
        cell["code_template"] = tmpl
        modified = True

    call_info = parse_call_expression(tmpl)
    if call_info:
        func_expr, raw_args, raw_kwargs = call_info
        func_obj = resolve_callable_from_expr(func_expr, cell.get("dependencies"))

        if func_obj is not None and callable(func_obj):
            reconstructed = validate_and_reconstruct_call(func_obj, func_expr, raw_args, raw_kwargs)
            new_tmpl = f"{{output_var}} = {reconstructed}"

            if new_tmpl != tmpl:
                cell["code_template"] = clean_malformed_template_braces(new_tmpl)
                modified = True

    # Enforce wiring invariant (placeholder sets vs declared input ports)
    if repair_wiring_invariant(cell, domain):
        modified = True

    return modified
