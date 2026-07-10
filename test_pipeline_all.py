import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from router import HardwareProfiler
from inference import ModelManager

def run_test(profile: str, embedder: str, llm: str, embedder_device: str, llm_device: str):
    print(f"\n==================================================")
    print(f" TESTING PROFILE {profile} | Emb: {embedder_device.upper()} | LLM: {llm_device.upper()}")
    print(f"==================================================")
    
    HardwareProfiler.set_config(embedder_device, llm_device, "ram")
    
    manager = ModelManager.get_instance()
    
    try:
        t0 = time.time()
        manager.initialize_profile(profile, embedder, llm)
        print(f"  [+] Initialized Profile {profile} in {time.time() - t0:.2f}s")
        
        # Test Embedding
        if profile in ["A", "C"]:
            emb = manager.get_embedding("Hello world, this is a test embedding.")
            print(f"  [+] Generated embedding of dimension {len(emb)}")
        
        # Test Generation
        if profile in ["B", "C"]:
            prompt = "Write a simple Python function to add two numbers."
            print(f"  [>] Prompt: {prompt}")
            gen = manager.generate_text(prompt, max_tokens=100)
            print(f"  [+] Generated text ({len(gen)} chars):\n{gen[:100]}...")
            
            # Test Feedback Check
            print(f"  [>] Running Feedback Check")
            fback = manager.feedback_check(gen)
            print(f"  [+] Feedback check passed. Length: {len(fback)}")
        else:
            print(f"  [-] Skipping generation for Profile {profile} (not supported).")
            
        print(f"  [SUCCESS] Profile {profile} passed.")
    except Exception as e:
        import traceback
        print(f"  [FAILED] Profile {profile} crashed!")
        traceback.print_exc()


if __name__ == "__main__":
    emb_model = "jina-embeddings-v5-text-nano"
    llm_1 = "qwen2.5-coder-0.5b-instruct"
    llm_2 = "qwen2.5-coder-1.5b-instruct"

    # Profile A (Embedding Only)
    run_test("A", emb_model, None, "cpu", "cpu")
    run_test("A", emb_model, None, "cuda", "cpu")
    
    # Profile B (LLM Only) - 0.5B
    run_test("B", emb_model, llm_1, "cpu", "cuda")
    # Profile B (LLM Only) - 1.5B
    run_test("B", emb_model, llm_2, "cpu", "cuda")

    # Profile C (Hybrid) - 0.5B
    run_test("C", emb_model, llm_1, "cuda", "cpu")
    # Profile C (Hybrid) - 1.5B
    run_test("C", emb_model, llm_2, "cuda", "cuda")
