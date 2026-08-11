import json
import logging
import re
from typing import Dict, Any

from log_config import get_logger
logger = get_logger("planner")

from lattice import LatticeOrchestrator, MicroCell
from inference import ModelManager

class ZeroShotPlanner:
    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine=None):
        self.orchestrator = orchestrator
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
            context_str = "\n".join(available_nodes)
            
        # Truncate context string to avoid blowing up the context window.
        # ~3000 tokens is roughly 12000 characters.
        if len(context_str) > 12000:
            logger.warning("Available context extremely large; truncating to fit within 4096 tokens.")
            context_str = context_str[:12000] + "\n... (truncated)"
        return context_str
        
    def _validate_sub_cells(self, macro_dict: dict) -> bool:
        available_ids = {cell.cell_id for cell in self.orchestrator.get_all_available_cells()}
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
            for i, sub_id in enumerate(sub_cells):
                if sub_id not in available_ids and not sub_id.startswith("SYNTH_"):
                    logger.warning(f"MISSING_NODE: LLM hallucinated node '{sub_id}'. Auto-correcting to SYNTH_{sub_id.upper()}")
                    sub_cells[i] = f"SYNTH_{sub_id.upper()}"
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
                and (cell.inputs.type_name == current_type or cell.inputs.type_name == "any")
                and (cell.inputs.state == current_state or cell.inputs.state in ("input_var", "source_identifier"))
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
        trunc_prompt = prompt[:100] + ("..." if len(prompt) > 100 else "")
        logger.info(f"Starting run_planning_pass with prompt: {trunc_prompt}")
        logger.info("Executing inference with JSON constraint via ModelManager...")

        if not ModelManager.get_instance().can_synthesize():
            logger.info("Active profile has no text generator; using deterministic planner fallback.")
            return self._run_deterministic_planning_pass(prompt)
        
        available_context = self._get_available_micro_nodes_context(prompt)
        num_nodes = available_context.count("- ID:")
        logger.debug(f"Available context summary: {num_nodes} nodes provided to LLM.")

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
      "sub_cells": ["<node_id_1>", "<node_id_2>"],
      "internal_topology": {{ "<source_id>": ["<dest_id>"] }}
    }}
  ]
}}

Available Micro-Nodes for your sub_cells:
{available_context}

CRITICAL INSTRUCTION: Your `sub_cells` array should contain a sequence of node IDs that accomplish the user request.
You must choose from the Available Micro-Nodes IF a suitable node exists. 
IF a specific step is required but no suitable micro-node exists in the available list, you MUST invent a new logical node ID starting with 'SYNTH_' (e.g., 'SYNTH_CALCULATE_SUM', 'SYNTH_EXTRACT_JSON'). The engine will dynamically synthesize this node at runtime."""

        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
        
        try:
            result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=2048)
        except Exception as e:
            raise RuntimeError(f"Planner LLM generation failed: {e}") from e
        
        logger.debug(f"Raw LLM output (first 200 chars): {result_text[:200]}")
        
        # Clean markdown code blocks if the LLM ignored the instruction
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[-1]
        if clean_text.endswith("```"):
            clean_text = clean_text.rsplit("```", 1)[0]
        clean_text = clean_text.strip()

        # 4. Parse and Validate
        try:
            macro_json = json.loads(clean_text)
            logger.info("Successfully parsed LLM output as JSON.")
            if not isinstance(macro_json, dict) or ("cells" not in macro_json and not isinstance(macro_json, list)):
                logger.warning("LLM returned unexpected JSON structure (missing 'cells' or not a dict/list). Using fallback.")
                return self._run_deterministic_planning_pass(prompt)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM output: {e}. Falling back to deterministic planner pass.")
            return self._run_deterministic_planning_pass(prompt)

        # BUG 13 FIX: Do NOT inject cells when validation fails.
        # Previously, malformed cells with hallucinated sub-cell IDs were injected
        # anyway, permanently polluting loaded_cells with broken macro nodes.
        if not self._validate_sub_cells(macro_json):
            logger.warning(
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
