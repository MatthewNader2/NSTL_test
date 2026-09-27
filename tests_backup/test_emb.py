import os
import unittest
from pathlib import Path

class TestEmb(unittest.TestCase):
    def test_embedding(self):
        model_path = "models/llms/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        if not os.path.exists(model_path):
            self.skipTest(f"Model path {model_path} not found")
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=512, embedding=True, verbose=False)
        t = "ID: numpy_seed_1 | Keywords: math array | Flow: any[any] -> any[any]"
        raw = llm.create_embedding(t)
        emb = raw['data'][0]['embedding']
        self.assertIsNotNone(emb)

if __name__ == "__main__":
    unittest.main()
