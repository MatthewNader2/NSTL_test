import os
import sys
import json
import logging

sys.path.insert(0, os.getcwd())

from inference import ModelManager
from internal_rag import LocalRAG

TREES_DIR = "trees"
MACRO_DIR = os.path.join(TREES_DIR, "macro")

MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "type": {"type": "string", "enum": ["macro"]},
                    "stage": {"type": "integer"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "inputs": {
                        "type": "object",
                        "properties": {
                            "type_name": {"type": "string"},
                            "state": {"type": "string"}
                        },
                        "required": ["type_name", "state"]
                    },
                    "outputs": {
                        "type": "object",
                        "properties": {
                            "type_name": {"type": "string"},
                            "state": {"type": "string"}
                        },
                        "required": ["type_name", "state"]
                    },
                    "algorithmic_steps": {"type": "array", "items": {"type": "string"}},
                    "sub_cells": {"type": "array", "items": {"type": "string"}},
                    "internal_topology": {
                        "type": "object"
                    }
                },
                "required": ["cell_id", "type", "stage", "keywords", "inputs", "outputs", "algorithmic_steps", "sub_cells", "internal_topology"]
            }
        }
    },
    "required": ["cells"]
}

def generate_test():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestGenerator")
    
    os.makedirs(MACRO_DIR, exist_ok=True)
    
    ModelManager.get_instance().initialize_profile("B")
    rag = LocalRAG(trees_dir=TREES_DIR)
    
    concept = "A* Pathfinding"
    logger.info(f"Generating Macro-Node for concept: {concept}")
    
    rag_context = rag.get_relevant_context(f"Implementation details for {concept}", top_k=3)
    
    system_prompt = f"""You are an expert Software Architect defining abstract logic graphs.
Your task is to implement the algorithmic logic for: {concept}.

You must output a strictly formatted JSON object matching the MacroCell Schema.

Available Scraped Micro-Nodes (Selection Pool):
{rag_context}

RULES FOR SUB_CELLS:
1. ALWAYS try to use the IDs of the scraped Micro-Nodes listed above if they fit the algorithmic step.
2. If an algorithmic step requires functionality that is MISSING from the scraped pool, you are AUTHORIZED to invent a clean, semantic placeholder ID (e.g., 'micro_custom_sort_step').
"""
    user_prompt = f"Write the Macro-Node JSON for {concept}."
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    logger.info(f"Invoking LLM for {concept}...")
    try:
        result_text = ModelManager.get_instance().generate_text(
            prompt=full_prompt, 
            max_tokens=2048, 
            schema=MACRO_SCHEMA
        )
        print("RESULT:")
        print(result_text)
        macro_json = json.loads(result_text)
        logger.info(f"Successfully generated!")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        
if __name__ == "__main__":
    generate_test()
