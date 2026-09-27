import os
import unittest

class TestEmb2(unittest.TestCase):
    def test_embedding2(self):
        model_path = "model_2/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
        if not os.path.exists(model_path):
            self.skipTest(f"Model path {model_path} not found")
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, embedding=True, verbose=False)
        result = llm.create_embedding("hello world")
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
