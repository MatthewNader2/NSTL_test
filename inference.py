import os
import time
import threading
import psutil
import logging
from abc import ABC, abstractmethod
from typing import List

# Mock transformers.onnx for jinaai embedding models compatibility
import sys
if 'transformers.onnx' not in sys.modules:
    import types
    transformers_onnx = types.ModuleType('transformers.onnx')
    transformers_onnx.OnnxConfig = type('OnnxConfig', (object,), {})
    sys.modules['transformers.onnx'] = transformers_onnx

import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'find_pruneable_heads_and_indices'):
    def find_pruneable_heads_and_indices(*args, **kwargs):
        return set(), []
    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

import transformers.configuration_utils
if not hasattr(transformers.configuration_utils.PretrainedConfig, 'is_decoder'):
    transformers.configuration_utils.PretrainedConfig.is_decoder = False
if not hasattr(transformers.configuration_utils.PretrainedConfig, 'add_cross_attention'):
    transformers.configuration_utils.PretrainedConfig.add_cross_attention = False

# Setup benchmarking logger
bench_logger = logging.getLogger("Benchmark")
bench_logger.setLevel(logging.INFO)
if not bench_logger.handlers:
    fh = logging.FileHandler("benchmark_metrics.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    bench_logger.addHandler(fh)

class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self, embedder_name: str, llm_name: str):
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        pass

    @abstractmethod
    def can_synthesize(self) -> bool:
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass
        
    @abstractmethod
    def feedback_check(self, generated_code: str) -> str:
        '''Runs a feedback check on the generated code (for Profiles B and C).'''
        pass


class BenchmarkProfile_A(InferenceProfile):
    def __init__(self):
        self.model = None

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from router import HardwareProfiler
        
        if not embedder_name:
            embedder_name = "jinaai/jina-embeddings-v2-small-en"
            
        model_path = os.path.join(os.getcwd(), "models", "embeddings", embedder_name)
        if not os.path.exists(model_path):
            model_path = embedder_name  # Fallback to HuggingFace hub if local missing
            
        device = HardwareProfiler.get_embedder_device()
        if device == "mps":
             device = "cpu" # SentenceTransformers MPS support is flaky
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except TypeError:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        raise RuntimeError("Profile A (Embedding Only) does not support text generation.")

    def can_synthesize(self) -> bool:
        return False
        
    def feedback_check(self, generated_code: str) -> str:
        return generated_code  # No feedback loop in Profile A

    @property
    def embedding_dimension(self) -> int:
        return 768


class BenchmarkProfile_B(InferenceProfile):
    def __init__(self):
        self.llm = None
        self._dim = 896

    def load_models(self, embedder_name: str, llm_name: str):
        from llama_cpp import Llama
        from router import HardwareProfiler
        
        if not llm_name:
            raise ValueError("Profile B requires an llm_name")
            
        model_dir = os.path.join(os.getcwd(), "models", "llms", llm_name)
        # Find the .gguf file inside the directory
        gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')] if os.path.exists(model_dir) else []
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf model found in {model_dir}")
        model_path = os.path.join(model_dir, gguf_files[0])

        device = HardwareProfiler.get_llm_device()
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
        # Attempt to auto-detect dimension if possible
        try:
            self._dim = self.llm.metadata.get('llama.embedding_length', 896)
            if isinstance(self._dim, str):
                self._dim = int(self._dim)
        except Exception:
            pass

    def get_embedding(self, text: str) -> List[float]:
        # Halting generation to do embedding (Mid-Processing)
        result = self.llm.create_embedding(text)
        return result['data'][0]['embedding']

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stop": ["```\\n", "}\\n", "}\\r\\n"],
            "temperature": 0.1,
            "top_p": 0.95
        }
        if schema:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema
            }
        
        response = self.llm(**kwargs)
        return response['choices'][0]['text'].strip()

    def can_synthesize(self) -> bool:
        return True
        
    def feedback_check(self, generated_code: str) -> str:
        prompt = f"Review the following generated python code for syntax errors or logical bugs. If there are errors, return the corrected code. Otherwise return the original code exactly. Only output the code:\\n\\n```python\\n{generated_code}\\n```"
        corrected = self.generate_text(prompt, max_tokens=1024)
        if "```python" in corrected:
            corrected = corrected.split("```python")[1].split("```")[0].strip()
        return corrected if corrected else generated_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_C(InferenceProfile):
    def __init__(self):
        self.embedder = None
        self.llm = None
        self._dim = 768

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        from router import HardwareProfiler
        
        if not embedder_name:
             embedder_name = "jinaai/jina-embeddings-v2-small-en"
        if not llm_name:
             raise ValueError("Profile C requires an llm_name")
             
        # Load Embedder
        emb_path = os.path.join(os.getcwd(), "models", "embeddings", embedder_name)
        if not os.path.exists(emb_path):
            emb_path = embedder_name
            
        emb_device = HardwareProfiler.get_embedder_device()
        if emb_device == "mps": emb_device = "cpu"
        self.embedder = SentenceTransformer(emb_path, device=emb_device, trust_remote_code=True, model_kwargs={'low_cpu_mem_usage': False})
        
        # Load LLM
        llm_dir = os.path.join(os.getcwd(), "models", "llms", llm_name)
        gguf_files = [f for f in os.listdir(llm_dir) if f.endswith('.gguf')] if os.path.exists(llm_dir) else []
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf model found in {llm_dir}")
        llm_path = os.path.join(llm_dir, gguf_files[0])
        
        llm_device = HardwareProfiler.get_llm_device()
        gpu_layers = -1 if llm_device == "cuda" else 0
        self.llm = Llama(
            model_path=llm_path,
            n_ctx=4096,
            mmap=True,
            verbose=False,
            n_gpu_layers=gpu_layers,
            embedding=False
        )

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedder.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except TypeError:
            return self.embedder.encode([text], convert_to_numpy=True)[0].tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stop": ["```\n", "}\n", "}\r\n"],
            "temperature": 0.1,
            "top_p": 0.95
        }
        if schema:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema
            }
        
        response = self.llm(**kwargs)
        return response['choices'][0]['text'].strip()

    def can_synthesize(self) -> bool:
        return True
        
    def feedback_check(self, generated_code: str) -> str:
        prompt = f"Review the following generated python code for syntax errors or logical bugs. If there are errors, return the corrected code. Otherwise return the original code exactly. Only output the code:\n\n```python\n{generated_code}\n```"
        corrected = self.generate_text(prompt, max_tokens=1024)
        if "```python" in corrected:
            corrected = corrected.split("```python")[1].split("```")[0].strip()
        return corrected if corrected else generated_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        if ModelManager._instance is not None:
            raise Exception("Singleton class. Use get_instance().")
        self.active_profile: InferenceProfile = None
        self.current_profile_name = None
        self.benchmarking_enabled = False

    @property
    def profile(self):
        return self.active_profile

    @staticmethod
    def get_instance():
        if ModelManager._instance is None:
            with ModelManager._lock:
                if ModelManager._instance is None:
                    ModelManager._instance = ModelManager()
        return ModelManager._instance

    def initialize_profile(self, profile_type: str, embedder_name: str = None, llm_name: str = None):
        with self._lock:
            # Re-initialize if the models change, even if the profile type is the same
            logging.info(f"Loading BenchmarkProfile_{profile_type} models persistently...")
            t0 = time.time()
            if profile_type == "A":
                new_profile = BenchmarkProfile_A()
            elif profile_type == "B":
                new_profile = BenchmarkProfile_B()
            elif profile_type == "C":
                new_profile = BenchmarkProfile_C()
            else:
                raise ValueError(f"Unknown profile: {profile_type}")
                
            new_profile.load_models(embedder_name, llm_name)
            
            self.active_profile = new_profile
            self.current_profile_name = profile_type
            
            t1 = time.time()
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            bench_logger.info(f"[Initialized Profile_{profile_type}] System RAM Usage: {mem:.2f} MB")
            print(f"  [+] Initialized Profile {profile_type} in {t1-t0:.2f}s")

    def get_embedding(self, text: str) -> List[float]:
        if self.active_profile is None: return []
        t0 = time.time()
        emb = self.active_profile.get_embedding(text)
        t1 = time.time()
        if self.benchmarking_enabled:
            bench_logger.info(f"[Embedding] Latency: {t1-t0:.4f}s")
        return emb

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.active_profile is None: return []
        return self.active_profile.get_embeddings(texts)

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        if self.active_profile is None: return ""
        if not self.active_profile.can_synthesize():
            logging.info("Active profile has no text generator; using deterministic planner fallback.")
            return ""
        t0 = time.time()
        text = self.active_profile.generate_text(prompt, max_tokens, schema)
        t1 = time.time()
        if self.benchmarking_enabled:
            bench_logger.info(f"[Generation] Latency: {t1-t0:.4f}s")
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            bench_logger.info(f"[Post-Generation] System RAM Usage: {mem:.2f} MB")
        return text

    def can_synthesize(self) -> bool:
        if self.active_profile is None: return False
        return self.active_profile.can_synthesize()
        
    def feedback_check(self, generated_code: str) -> str:
        if self.active_profile is None: return generated_code
        return self.active_profile.feedback_check(generated_code)

    @property
    def embedding_dimension(self) -> int:
        if self.active_profile is None: return 0
        return self.active_profile.embedding_dimension
