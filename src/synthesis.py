"""
src/synthesis.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamic MicroCell Synthesizer: Grounded in Live API Documentation.
"""

from __future__ import annotations
import ast
import json
import os
import re
from typing import Dict, Any, Optional
from log_config import get_logger
from external_rag import LiveDocFetcher
from inference import ModelManager

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
        domain: str = "Python_Core"
    ) -> Dict[str, Any]:
        """
        Queries official API documentation and uses the LLM to synthesize
        a verified, type-annotated MicroCell.
        """
        logger.info(f"[SYNTHESIS] Fetching live documentation for: '{gap_concept}'")
        live_docs = fetcher.fetch(gap_concept) or "No live documentation available."

        system_prompt = f"""You are an expert Software Engineer. Synthesize a single verified Python computational node implementing: '{gap_concept}'.
Output ONLY a valid JSON object matching the schema below. No markdown formatting, no explanations.

Schema:
{{
  "cell_id": "micro_synthesized_{re.sub(r'[^a-zA-Z0-9_]', '_', gap_concept).lower()[:30]}",
  "type": "micro",
  "stage": 2,
  "keywords": ["{gap_concept}"],
  "inputs": {{ "type_name": "{expected_input}", "state": "any" }},
  "outputs": {{ "type_name": "{expected_output}", "state": "computed" }},
  "dependencies": ["<import statement>"],
  "code_template": "{{output_var}} = <function_call>({{input_var}})"
}}

Documentation Reference:
{live_docs[:1500]}

RULES:
1. The `code_template` MUST use `{{output_var}}` for output assignment and `{{input_var}}` for the input argument.
2. Put all necessary import statements in the `dependencies` array (e.g. ["import pandas as pd", "import cv2"]).
3. Ensure the code is strictly non-interactive (do not use input() or sys.stdin)."""

        full_prompt = f"{system_prompt}\n\nTask: {gap_concept}"

        result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=1024)

        # Clean markdown formatting
        clean_json = result_text.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[-1]
        if clean_json.endswith("```"):
            clean_json = clean_json.rsplit("```", 1)[0]
        clean_json = clean_json.strip()

        try:
            cell_dict = json.loads(clean_json)
        except json.JSONDecodeError as e:
            logger.error(f"[SYNTHESIS ERROR] Invalid JSON from model: {e}")
            raise ValueError(f"Model failed to generate valid JSON for {gap_concept}") from e

        # Validate template syntax by substituting dummy identifiers
        template = cell_dict.get("code_template", "")
        test_code = template.replace("{output_var}", "out_var").replace("{input_var}", "in_var")
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            logger.error(f"[SYNTHESIS ERROR] Synthesized code has syntax errors: {e}\n{test_code}")
            raise ValueError(f"Synthesized template for {gap_concept} failed AST parse check.") from e

        return cell_dict
