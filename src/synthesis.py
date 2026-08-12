import json
from log_config import get_logger
from typing import Dict

from lattice import MicroCell
from external_rag import LiveDocFetcher
from inference import ModelManager

logger = get_logger('synthesis')

class SynthesisEngine:
    def __init__(self):
        pass

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
        logger.info(f"Fetching live docs for concept: {gap_concept}")
        live_docs = fetcher.fetch(gap_concept)
        if not live_docs:
            logger.warning(f"No live docs found for {gap_concept}. Proceeding with zero-shot knowledge.")
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
      "code": "import math\\n{{output_var}} = math.sqrt(float({{input_var}}))",
      "dependencies": ["math"]
    }}
  }}
}}

CRITICAL INSTRUCTION: The `code` field MUST contain ACTUAL, functional Python code that implements the concept. DO NOT output placeholder text like `<library>` or `...({{input_var}})`. You MUST replace them with real library names (like `import json`) and real logic (like `a + b` or `json.loads(a)`).
CRITICAL INSTRUCTION 2: You MUST use the exact placeholder strings `{{input_var}}` and `{{output_var}}` in your python logic for the main input and output of the cell. The engine will dynamically replace these at runtime. Do NOT use hardcoded variable names for the input or output.

Use the following Official Documentation as your absolute ground truth:"""
        prompt = f"{system_prompt}\n\nTask: {gap_concept}\nLive Context:\n{live_docs}"

        logger.info("Executing generation via ModelManager...")
        result_text = ModelManager.get_instance().generate_text(prompt, max_tokens=1024)
        
        cleaned_text = result_text.strip()
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()

        # 3. Validation
        try:
            micro_json = json.loads(cleaned_text)
            return micro_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Synthesis LLM output: {e}\nRaw output:\n{result_text[:500]}")
            raise ValueError(f"Synthesized output was not valid JSON. Raw output: {result_text[:200]}")
