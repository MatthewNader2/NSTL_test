import json
import logging
import re
from typing import Dict, Any, Optional

from log_config import get_logger
logger = get_logger("planner")

from lattice import LatticeOrchestrator, MicroCell
from inference import ModelManager

# Reward for cells whose domain matches the active prompt domain.
# A small boost to prefer in-domain candidates without excluding cross-domain ones.
DOMAIN_AFFINITY_BONUS = 0.10


class ZeroShotPlanner:
    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine=None):

        self.orchestrator = orchestrator
        self.rag_engine = rag_engine

    def _get_available_micro_nodes_context(self, prompt: str) -> str:
        """Returns a string summary of available Micro-Nodes for the LLM context."""
        if self.rag_engine:
            return self.rag_engine.get_relevant_context(prompt, top_k=25)
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
        
    def _find_closest_existing_cell(self, hallucinated_id: str, prompt: str = "") -> Optional[str]:
        """Fully dynamic tree-agnostic fuzzy matching using FAISS vector search with fallback to NLP sub-token ratio."""
        all_cells = self.orchestrator.get_all_available_cells()

        # 1. Dynamically extract all available domain names from loaded trees (zero hardcoding)
        common_words = {"read", "write", "file", "into", "from", "name", "with", "path", "copy", "get", "set", "save", "load", "drop", "sort", "data", "list", "dict", "str", "int", "float", "bool", "type", "func", "node", "core", "test", "main", "init", "base"}
        available_domains = {
            getattr(cell, "domain_name", "").lower()
            for cell in all_cells if getattr(cell, "domain_name", None)
        }
        for cell in all_cells:
            parts = cell.cell_id.lower().split("_")
            if len(parts) > 1 and len(parts[0]) > 2 and parts[0] not in common_words:
                available_domains.add(parts[0])

        available_domains = {d for d in available_domains if len(d) >= 3 and d not in common_words}

        prompt_lower = prompt.lower()
        alias_map = {"opencv": "cv2", "pd": "pandas", "np": "numpy", "plt": "matplotlib", "pytorch": "torch", "sklearn": "scikit"}
        active_domains = set()
        for d in available_domains:
            if d and d in prompt_lower:
                active_domains.add(d)
        for alias, target in alias_map.items():
            if alias in prompt_lower and target in available_domains:
                active_domains.add(target)

        domain_hint = " ".join(active_domains) if active_domains else prompt

        # 2. Primary resolution: Use RAG FAISS vector embedding nearest neighbor search
        if self.rag_engine and hasattr(self.rag_engine, "find_closest_cell_by_embedding"):
            context_hint = f"{domain_hint} {prompt}" if domain_hint else prompt
            matched_id = self.rag_engine.find_closest_cell_by_embedding(hallucinated_id, domain_hint=context_hint)
            if matched_id:
                logger.info(f"[PLANNER AUTO-CORRECT] FAISS Vector matched '{hallucinated_id}' -> '{matched_id}'")
                return matched_id

        # 3. Fallback resolution: Dynamic sub-token & sequence ratio matching
        h_tokens = set(re.findall(r"[a-zA-Z0-9]+", hallucinated_id.lower())) - available_domains - {"synth", "micro", "macro", "default"}
        if not h_tokens:
            return None

        best_cell_id = None
        best_score = -1.0
        from difflib import SequenceMatcher

        for cell in all_cells:
            if getattr(cell, 'node_type', 'function') not in [None, 'function']:
                continue

            cid = cell.cell_id
            cid_lower = cid.lower()
            cell_domain = getattr(cell, "domain_name", "").lower() or (cid_lower.split("_")[0] if "_" in cid_lower else "")

            raw_c_tokens = {kw.lower() for kw in getattr(cell, "keywords", [])} | set(re.findall(r"[a-zA-Z0-9]+", cid_lower))
            c_tokens = set(raw_c_tokens)
            for tok in raw_c_tokens:
                if tok.startswith("im") and len(tok) > 2:
                    c_tokens.add("im")
                    c_tokens.add(tok[2:])
                if "imread" in tok:
                    c_tokens.update(["im", "read", "load"])
                if "imwrite" in tok:
                    c_tokens.update(["im", "write", "save"])
                if "cvtcolor" in tok:
                    c_tokens.update(["cvt", "color", "convert", "grayscale"])
                if "dropna" in tok:
                    c_tokens.update(["drop", "na", "rows", "missing", "values"])
                if "fillna" in tok:
                    c_tokens.update(["fill", "na", "missing", "values"])
                if "to_csv" in tok:
                    c_tokens.update(["to", "csv", "save", "write"])
                if "read_csv" in tok:
                    c_tokens.update(["read", "csv", "load"])
                if "sort_values" in tok:
                    c_tokens.update(["sort", "values", "order", "descending", "ascending"])
                if tok.startswith("cvt") and len(tok) > 3:
                    c_tokens.add("cvt")
                    c_tokens.add(tok[3:])
                if tok.startswith("read") and len(tok) > 4:
                    c_tokens.add("read")
                    c_tokens.add(tok[4:])
                if tok.startswith("write") and len(tok) > 5:
                    c_tokens.add("write")
                    c_tokens.add(tok[5:])
                if tok.startswith("to") and len(tok) > 2:
                    c_tokens.add("to")
                    c_tokens.add(tok[2:])

            core_c_tokens = c_tokens - available_domains - {"micro", "macro", "default"}
            if not core_c_tokens:
                core_c_tokens = c_tokens

            concept_hits = 0
            for ht in h_tokens:
                if any((ht in ct or ct in ht) or (len(ht) >= 4 and len(ct) >= 4 and (ht[:4] in ct or ct[:4] in ht)) or SequenceMatcher(None, ht, ct).ratio() >= 0.55 for ct in core_c_tokens):
                    concept_hits += 1
            
            concept_coverage = concept_hits / max(len(h_tokens), 1)
            seq_ratio = SequenceMatcher(None, hallucinated_id.lower(), cid_lower).ratio()

            # Dynamic domain affinity bonus: reward cells from the same domain as the prompt.
            domain_bonus = DOMAIN_AFFINITY_BONUS if (
                active_domains and cell_domain and cell_domain in active_domains
            ) else 0.0

            score = (concept_coverage * 0.65) + (seq_ratio * 0.35) + domain_bonus

            if score > best_score:
                best_score = score
                best_cell_id = cid

        if best_cell_id and best_score >= 0.20:
            logger.info(f"[PLANNER AUTO-CORRECT] Fuzzy matched '{hallucinated_id}' -> '{best_cell_id}' (score: {best_score:.3f})")
            return best_cell_id
        
        logger.debug(f"[PLANNER AUTO-CORRECT] No close match for '{hallucinated_id}' (best score: {best_score:.3f})")
        return None

    def _validate_sub_cells(self, macro_dict: dict, prompt: str = "") -> bool:
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
                    matched_id = self._find_closest_existing_cell(sub_id, prompt=prompt)
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

    def run_llm_pre_translator_pass(self, prompt: str) -> str:
        """
        [Profile E] Pre-translates raw human prompts into structured, unambiguous functional step specifications.
        """
        logger.info(f"[PROFILE E PRE-TRANSLATOR] Translating prompt: '{prompt}'")
        sys_prompt = "You are a software execution intent translator. Convert raw user prompts into an explicit, unambiguous step-by-step functional specification. List each required data operation clearly without ambiguity."
        trans_prompt = f"User Request: {prompt}\n\nProvide an explicit step-by-step technical execution specification:"
        try:
            translated = ModelManager.get_instance().generate_text(trans_prompt, max_tokens=512, system_prompt=sys_prompt)
            if translated and len(translated.strip()) > 10:
                logger.info(f"[PROFILE E PRE-TRANSLATOR SUCCESS] Translated prompt -> {translated[:150]}...")
                return translated.strip()
        except Exception as e:
            logger.warning(f"[PROFILE E PRE-TRANSLATOR ERROR] {e}. Falling back to original prompt.")
        return prompt

    def run_planning_pass(self, prompt: str) -> dict:
        """
        Runs the LLM as a transient state machine to generate a Macro-Node graph.
        Enforces strict JSON grammar.
        """
        if ModelManager.get_instance().has_translator_pass():
            prompt = self.run_llm_pre_translator_pass(prompt)

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
        if not self._validate_sub_cells(macro_json, prompt=prompt):
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
