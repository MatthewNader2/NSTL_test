import sys
with open('inference.py', 'r', encoding='utf-8') as f:
    code = f.read()

profile_c = '''
class BenchmarkProfile_C(BenchmarkProfile_B):
    def load_models(self):
        from llama_cpp import Llama
        from router import HardwareProfiler
        import os
        model_path = os.path.join(os.getcwd(), "model_3", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        device = HardwareProfiler.get_optimal_device()
        gpu_layers = -1 if device == "cuda" else 0
        common_kwargs = {
            "model_path": model_path,
            "n_ctx": 4096,
            "mmap": True,
            "verbose": False,
            "n_gpu_layers": gpu_layers,
            "embedding": True,
        }
        self.llm = Llama(**common_kwargs)

    @property
    def embedding_dimension(self) -> int:
        return 1536

class ModelManager:'''

code = code.replace('class ModelManager:', profile_c)

patch = '''            if profile_type == "A":
                new_profile = BenchmarkProfile_A()
            elif profile_type == "B":
                new_profile = BenchmarkProfile_B()
            elif profile_type == "C":
                new_profile = BenchmarkProfile_C()
            else:'''

code = code.replace('''            if profile_type == "A":
                new_profile = BenchmarkProfile_A()
            elif profile_type == "B":
                new_profile = BenchmarkProfile_B()
            else:''', patch)

with open('inference.py', 'w', encoding='utf-8') as f:
    f.write(code)
