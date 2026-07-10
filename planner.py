import json
import logging
import re
from typing import Dict, Any

from lattice import LatticeOrchestrator, MicroCell
from inference import ModelManager

class ZeroShotPlanner:
    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine=None):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("ZeroShotPlanner")
        self.rag_engine = rag_engine

    def _get_available_micro_nodes_context(self, prompt: str) -> str:
        """Returns a string summary of available Micro-Nodes for the LLM context."""
        if self.rag_engine:
            return self.rag_engine.get_relevant_context(prompt, top_k=15)
        else:
            # Fallback if no RAG is provided
            available_nodes = []
            for cell in self.orchestrator.get_all_available_cells():
                if isinstance(cell, MicroCell):
                    desc = (
                        f"- ID: {cell.cell_id} | "
                        f"Inputs: {cell.inputs.type_name}[{cell.inputs.state}] -> "
                        f"Outputs: {cell.outputs.type_name}[{cell.outputs.state}]"
                    )
                    available_nodes.append(desc)
            return "\n".join(available_nodes)
        
    def _validate_sub_cells(self, macro_dict: dict) -> bool:
        """
        Pre-Injection Validation: Checks if the LLM hallucinated a Micro-Node ID.
        """
        available_ids = {c.cell_id for c in self.orchestrator.get_all_available_cells() if isinstance(c, MicroCell)}
        
        # In the schema, macro_dict might have multiple cells if it's the root JSON, 
        # but we expect macro_dict to be the specific cell dict.
        # Let's handle both in case the LLM wrapped it in {"cells": [...]}
        cells_to_check = []
        if "cells" in macro_dict:
            cells_to_check = macro_dict["cells"]
        else:
            cells_to_check = [macro_dict]

        for cell_dict in cells_to_check:
            if not isinstance(cell_dict, dict):
                return False
            sub_cells = cell_dict.get("sub_cells", [])
            if not isinstance(sub_cells, list):
                return False
            for sub_id in sub_cells:
                if sub_id not in available_ids:
                    self.logger.warning(f"MISSING_NODE: LLM hallucinated node '{sub_id}'")
                    return False
        return True

    def _run_deterministic_planning_pass(self, prompt: str) -> dict:
        """
        Builds a simple executable macro from existing micro-cells when the active
        profile has no text generator. It favors prompt keyword matches while
        preserving type/state continuity from the default input source.
        """
        tokens = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))
        all_micro = [
            cell for cell in self.orchestrator.get_all_available_cells()
            if isinstance(cell, MicroCell)
        ]
        current_type = "str"
        current_state = "source_identifier"
        selected = []
        used_ids = set()

        for _ in range(8):
            compatible = [
                cell for cell in all_micro
                if cell.cell_id not in used_ids
                and cell.inputs.type_name == current_type
                and cell.inputs.state == current_state
            ]
            if not compatible:
                break

            scored = []
            for cell in compatible:
                keyword_hits = tokens.intersection({kw.lower() for kw in cell.keywords})
                id_hits = {
                    part for part in re.split(r"[_\W]+", cell.cell_id.lower())
                    if part and part in tokens
                }
                score = (2 * len(keyword_hits)) + len(id_hits)
                if score:
                    scored.append((score, cell.stage, cell.cell_id, cell))

            if not scored:
                break

            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            next_cell = scored[0][3]
            selected.append(next_cell.cell_id)
            used_ids.add(next_cell.cell_id)
            current_type = next_cell.outputs.type_name
            current_state = next_cell.outputs.state

        if not selected:
            raise ValueError("Deterministic planner could not find an executable path for this prompt.")

        return {
            "cells": [
                {
                    "cell_id": "macro_deterministic_fallback",
                    "type": "macro",
                    "stage": 1,
                    "keywords": sorted(tokens)[:12],
                    "inputs": {"type_name": "str", "state": "source_identifier"},
                    "outputs": {"type_name": current_type, "state": current_state},
                    "algorithmic_steps": selected,
                    "sub_cells": selected,
                    "internal_topology": {
                        selected[i]: [selected[i + 1]]
                        for i in range(len(selected) - 1)
                    },
                }
            ]
        }

    def run_planning_pass(self, prompt: str) -> dict:
        """
        Runs the LLM as a transient state machine to generate a Macro-Node graph.
        Enforces strict JSON grammar.
        """
        self.logger.info("Executing inference with JSON constraint via ModelManager...")

        if not ModelManager.get_instance().can_synthesize():
            self.logger.info("Active profile has no text generator; using deterministic planner fallback.")
            return self._run_deterministic_planning_pass(prompt)
        
        available_context = self._get_available_micro_nodes_context(prompt)

        system_prompt = f"""You are an expert Software Architect. You decompose complex user requests into a strict topological graph of computational steps.
You MUST output ONLY valid JSON matching the following schema. No markdown formatting, no explanations.

Schema:
{{
  "cells": [
    {{
      "cell_id": "macro_dynamic_<task_name>",
      "type": "macro",
      "stage": 1,
      "keywords": ["..."],
      "inputs": {{ "type_name": "...", "state": "..." }},
      "outputs": {{ "type_name": "...", "state": "..." }},
      "algorithmic_steps": ["1. ...", "2. ..."],
      "sub_cells": ["<must_be_chosen_from_available_micro_nodes>"],
      "internal_topology": {{ "<source_id>": ["<dest_id>"] }}
    }}
  ]
}}

Available Micro-Nodes for your sub_cells:
{available_context}"""

        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
        
        result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=2048)
        
        # 4. Parse and Validate
        try:
            macro_json = json.loads(result_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM output: {e}")
            raise ValueError("LLM output was not valid JSON.")

        # BUG 13 FIX: Do NOT inject cells when validation fails.
        # Previously, malformed cells with hallucinated sub-cell IDs were injected
        # anyway, permanently polluting loaded_cells with broken macro nodes.
        if not self._validate_sub_cells(macro_json):
            self.logger.warning(
                "Planner returned hallucinated node IDs. Injection aborted. "
                "The missing nodes will need to be synthesized at runtime."
            )
            # Return the macro_json so the caller can still iterate sub_cells
            # and trigger synthesis for the missing ones — but do not inject
            # the malformed macro itself into the orchestrator.
            return macro_json

        # If it wrapped it in "cells": [], unwrap it for injection
        cells_to_inject = macro_json.get("cells", [macro_json])
        
        for cell_dict in cells_to_inject:
            self.orchestrator.inject_transient_macro(cell_dict)
            
        return macro_json
