"""
src/utils.py - Shared utilities for NSTL
Extracts common patterns to eliminate code duplication.
"""

from __future__ import annotations
import ast
import json
import re
from typing import Any, Dict, Optional, Set
from log_config import get_logger

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
        text = re.sub(r"^```(?:json|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
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
    test_code = template
    for ph in set(re.findall(r"\{([a-zA-Z0-9_]+)\}", test_code)):
        test_code = test_code.replace(f"{{{ph}}}", f"_ph_{ph}")
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False
