import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import find_llama_server, MODELS_DIR

class TestApi2(unittest.TestCase):
    def test_api2(self):
        server_exe = find_llama_server()
        model_path = os.path.join(MODELS_DIR, "llms", "Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")

        if not server_exe or not os.path.exists(server_exe) or not os.access(server_exe, os.X_OK):
            self.skipTest(f"llama-server executable not found or not executable ({server_exe})")

        if not os.path.exists(model_path):
            self.skipTest(f"GGUF model not found ({model_path})")

if __name__ == "__main__":
    unittest.main()
