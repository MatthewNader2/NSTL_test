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
from utils import extract_json_from_llm, validate_code_template

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

        # Infer stage if not explicitly passed
        gc_lower = gap_concept.lower()
        if stage is None:
            if any(k in gc_lower for k in ("read", "load", "ingest", "from_")) or expected_input.lower() in ("str", "source_identifier"):
                stage = 1
            elif any(k in gc_lower for k in ("save", "write", "to_csv", "export", "to_")):
                stage = 3
            else:
                stage = 2

        if stage == 1:
            template_rule = "For Stage 1 (file loading/source), `code_template` MUST use `{filepath}` as the input path argument and `{output_var}` for the output assignment (e.g. `{output_var} = pd.read_csv({filepath})`)."
            input_spec = '{"type_name": "str", "state": "source_identifier"}'
            sample_template = "{output_var} = <library>.<read_func>({filepath})"
        elif stage == 3:
            template_rule = "For Stage 3 (exporting/saving), `code_template` MUST use `{dest_path}` as the destination file path (e.g. `{input_var}.to_csv({dest_path}, index=False)\\n{output_var} = {dest_path}`)."
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{input_var}.<save_func>({dest_path})\\n{output_var} = {dest_path}"
        else:
            template_rule = "For Stage 2 (data transform), `code_template` MUST use `{input_var}` (or `{df}`) for input and `{output_var}` for output (e.g. `{output_var} = <func>({input_var})`)."
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{output_var} = <library>.<func>({input_var})"

        system_prompt = f"""You are an expert Software Engineer. Synthesize a single verified Python computational node implementing: '{gap_concept}'.
Output ONLY a valid JSON object matching the schema below. No markdown formatting, no explanations.

Schema:
{{
  "cell_id": "micro_synthesized_{re.sub(r'[^a-zA-Z0-9_]', '_', gap_concept).lower()[:30]}",
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
2. Put all necessary import statements in the `dependencies` array (e.g. ["import pandas as pd", "import cv2"]).
3. Ensure the code is strictly non-interactive (do not use input() or sys.stdin).
4. Never hardcode data filenames like 'data.csv' — always use `{{filepath}}` or `{{dest_path}}` placeholders."""

        full_prompt = f"{system_prompt}\n\nTask: {gap_concept}"

        result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=1024)

        cell_dict = extract_json_from_llm(result_text)
        if cell_dict is None:
            logger.error(f"[SYNTHESIS ERROR] Failed to extract JSON from model output")
            raise ValueError(f"Model failed to generate valid JSON for {gap_concept}")

        # Validate template syntax by substituting dummy identifiers for all placeholders
        template = cell_dict.get("code_template", "")
        if not validate_code_template(template):
            logger.error(f"[SYNTHESIS ERROR] Synthesized template failed AST validation")
            raise ValueError(f"Synthesized template for {gap_concept} failed AST parse check.")

        return cell_dict
