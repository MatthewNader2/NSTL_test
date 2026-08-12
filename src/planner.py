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
            return self.rag_engine.get_relevant_context(prompt, top_k=10)
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
            
        if len(context_str) > 6000:
            context_str = context_str[:6000] + "\n... (truncated)"
        return context_str
        
    def _find_closest_existing_cell(self, hallucinated_id: str) -> str:
        h_tokens = set(re.findall(r"[a-zA-Z_]+", hallucinated_id.lower())) - {"pandas", "cv2", "numpy", "scikit", "torch"}
        if not h_tokens:
            return None

        best_cell_id = None
        best_score = -1.0

        for cell in self.orchestrator.get_all_available_cells():
            cid = cell.cell_id
            cid_lower = cid.lower()
            c_tokens = {kw.lower() for kw in getattr(cell, 'keywords', [])} | set(re.findall(r"[a-zA-Z_]+", cid_lower))
            overlap = len(h_tokens.intersection(c_tokens)) / max(len(h_tokens), 1)

            # Main operation matching
            h_str = "".join(h_tokens)
            if "dropna" in cid_lower and ("drop" in h_str or "na" in h_str or "null" in h_str or "missing" in h_str):
                overlap += 1.2
            elif "sort_values" in cid_lower and "sort" in h_str:
                overlap += 1.2
            elif "imread" in cid_lower and ("read" in h_str or "load" in h_str or "imread" in h_str) and "save" not in h_str and "write" not in h_str:
                overlap += 1.8
            elif "imwrite" in cid_lower and ("save" in h_str or "write" in h_str or "imwrite" in h_str or "export" in h_str):
                overlap += 1.8
            elif "cvtcolor" in cid_lower and ("convert" in h_str or "color" in h_str or "grayscale" in h_str or "gray" in h_str):
                overlap += 1.8
            elif "to_csv" in cid_lower and ("csv" in h_str or "save" in h_str or "write" in h_str) and "image" not in h_str:
                overlap += 1.2
            elif "read_csv" in cid_lower and ("csv" in h_str or "read" in h_str or "load" in h_str) and "image" not in h_str:
                overlap += 1.2
            elif cid_lower == "pandas_dataframe_drop" and "dropna" not in h_str:
                overlap -= 0.5

            if overlap > best_score:
                best_score = overlap
                best_cell_id = cid

        if best_match := (best_cell_id if best_score >= 0.6 else None):
            return best_match
        return None

    def _validate_sub_cells(self, macro_dict: dict) -> bool:
        available_ids = {cell.cell_id for cell in self.orchestrator.get_all_available_cells()}
        cells_to_check = macro_dict.get("cells", [macro_dict]) if isinstance(macro_dict, dict) else []

        for cell_dict in cells_to_check:
            if not isinstance(cell_dict, dict):
                return False
            sub_cells = cell_dict.get("sub_cells", [])
            if not isinstance(sub_cells, list):
                return False
            for i, sub_id in enumerate(sub_cells):
                if sub_id not in available_ids and not sub_id.startswith("SYNTH_"):
                    # Try fuzzy resolution to real cell in database first
                    matched_id = self._find_closest_existing_cell(sub_id)
                    if matched_id:
                        logger.info(f"[PLANNER AUTO-CORRECT] Resolved hallucinated '{sub_id}' -> '{matched_id}'")
                        sub_cells[i] = matched_id
                    else:
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

CRITICAL INSTRUCTION: Your `sub_cells` array should contain a MINIMAL, CONCISE sequence of node IDs (typically 2 to 5 steps) that accomplish the user request. DO NOT include extra, redundant, or repetitive steps after the goal is completed.
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
        macro_json = None
        try:
            macro_json = json.loads(clean_text)
            logger.info("Successfully parsed LLM output as JSON.")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}. Attempting regex recovery of sub_cells...")
            sub_matches = re.findall(r'["\'](PANDAS_[A-Z0-9_]+|CV2_[A-Z0-9_]+|NUMPY_[A-Z0-9_]+|SYNTH_[A-Z0-9_]+)["\']', clean_text)
            if sub_matches:
                logger.info(f"Regex recovered {len(sub_matches)} cell IDs: {sub_matches}")
                macro_json = {
                    "cells": [{
                        "cell_id": "macro_dynamic_recovered",
                        "type": "macro",
                        "stage": 1,
                        "keywords": [],
                        "inputs": {"type_name": "any", "state": "any"},
                        "outputs": {"type_name": "any", "state": "any"},
                        "sub_cells": sub_matches
                    }]
                }
            else:
                logger.warning("Regex recovery failed; falling back to deterministic planner pass.")
                return self._run_deterministic_planning_pass(prompt)

        if not isinstance(macro_json, dict) or ("cells" not in macro_json and not isinstance(macro_json, list)):
            logger.warning("LLM returned unexpected JSON structure. Using fallback.")
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
