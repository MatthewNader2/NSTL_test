import json
import logging
from typing import Dict

from lattice import MicroCell
from external_rag import LiveDocFetcher
from inference import ModelManager

class SynthesisEngine:
    def __init__(self):
        self.logger = logging.getLogger("SynthesisEngine")

    def synthesize_micro_cell(
        self, 
        gap_concept: str, 
        expected_input: str, 
        expected_output: str, 
        fetcher: LiveDocFetcher
    ) -> dict:
        """
        Dynamically generates a MicroCell to bridge a topological gap using live docs.
        Enforces strict VRAM/RAM purging and JSON grammar (A↑, R↓).
        """
        self.logger.info(f"Fetching live docs for concept: {gap_concept}")
        live_docs = fetcher.fetch(gap_concept)
        if not live_docs:
            self.logger.warning(f"No live docs found for {gap_concept}. Proceeding with zero-shot knowledge.")
            live_docs = "No external documentation available."


        system_prompt = f"""You are an expert Software Engineer. You must write a Phase 2 MicroCell JSON that implements the concept '{gap_concept}'.
You MUST output ONLY valid JSON matching the schema. No markdown, no explanations.

Schema:
{{
  "cell_id": "micro_synthesized_<concept>",
  "type": "micro",
  "stage": 1,
  "keywords": ["<concept>"],
  "inputs": {{ "type_name": "{expected_input}", "state": "raw" }},
  "outputs": {{ "type_name": "{expected_output}", "state": "computed" }},
  "domain_implementations": {{
    "Python_Core": {{
      "code": "import <library>\\n{{output_var}} = ...(<{{input_var}}>)",
      "dependencies": ["<library>"]
    }}
  }}
}}

Use the following Official Documentation as your absolute ground truth:"""
        prompt = f"{system_prompt}\n\nTask: {gap_concept}\nLive Context:\n{live_docs}"

        self.logger.info("Executing generation via ModelManager...")
        result_text = ModelManager.get_instance().generate_text(prompt, max_tokens=1024)
        
        # 3. Validation
        try:
            micro_json = json.loads(result_text)
            return micro_json
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Synthesis LLM output: {e}")
            raise ValueError("Synthesized output was not valid JSON.")
