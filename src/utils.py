"""
src/utils.py - Shared utilities for NSTL
Extracts common patterns to eliminate code duplication.
"""

from __future__ import annotations
import ast
import json
from typing import Any, Dict, Optional, Set
try:
    from log_config import get_logger
except ImportError:
    from .log_config import get_logger

logger = get_logger('utils')


def extract_json_from_llm(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts a JSON object from LLM output.
    
    Handles: markdown fences, leading prose, multiple JSON blocks.
    Used by: planner.py, synthesis.py, generate_trees.py, llm_harvester.py
    """
    if not raw:
        return None
    text = raw.strip()
    
    # Strip markdown code fences
    if text.startswith("```"):
        first_line_end = text.find("\n")
        if first_line_end != -1:
            text = text[first_line_end + 1 :]
        else:
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    
    # Attempt 1: Direct parse (best case — clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Find the outermost balanced braces
    depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                try:
                    return json.loads(text[start_idx:i + 1])
                except json.JSONDecodeError:
                    start_idx = None  # Try next top-level block
    
    logger.warning(f"[UTILS] Failed to extract JSON from LLM output ({len(text)} chars)")
    return None


def validate_code_template(template: str) -> bool:
    """Validates a code template by substituting placeholders with dummy identifiers.
    
    Used by: cli.py, schema.py, synthesis.py
    Returns True if the template parses as valid Python after placeholder substitution.
    """
    if not template or not template.strip():
        return False
    
    placeholders: Set[str] = set()
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{" and i + 1 < n:
            end = template.find("}", i + 1)
            if end != -1:
                inner = template[i + 1 : end]
                if inner.isidentifier():
                    placeholders.add(inner)
                i = end + 1
                continue
        i += 1

    test_code = template
    for ph in placeholders:
        test_code = test_code.replace(f"{{{ph}}}", f"_ph_{ph}")
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False


def extract_code_from_llm_response(text: str) -> str:
    """Extracts executable Python code from an LLM response.

    Handles:
    - ```python\\n...\\n```
    - ```py\\n...\\n```
    - ```\\n...\\n``` (no language tag)
    - Plain code with no fences at all (return trimmed/unchanged)
    - Fenced code with leading/trailing prose outside the fence (extracts just the fenced block)
    """
    if not text:
        return ""

    start_idx = text.find("```")
    if start_idx != -1:
        newline_idx = text.find("\n", start_idx + 3)
        if newline_idx != -1:
            code_start = newline_idx + 1
        else:
            code_start = start_idx + 3
        end_idx = text.find("```", code_start)
        if end_idx != -1:
            return text[code_start:end_idx].strip()
        else:
            return text[code_start:].strip()

    return text.strip()

