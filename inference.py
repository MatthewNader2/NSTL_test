import os
import time
import threading
import psutil
import logging
from abc import ABC, abstractmethod
from typing import List

# Setup benchmarking logger
bench_logger = logging.getLogger("Benchmark")
bench_logger.setLevel(logging.INFO)
if not bench_logger.handlers:
    fh = logging.FileHandler("benchmark_metrics.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    bench_logger.addHandler(fh)

class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self):
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


class BenchmarkProfile_A(InferenceProfile):
    def __init__(self):
        self.model = None

    def load_models(self):
        from sentence_transformers import SentenceTransformer
        from router import HardwareProfiler
        model_path = os.path.join(os.getcwd(), "model_1")
        if not os.path.exists(model_path):
            model_path = "jinaai/jina-embeddings-v2-small-en"
        device = HardwareProfiler.get_optimal_device()
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except TypeError:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.encode(texts, convert_to_numpy=True, task="retrieval").tolist()
        except TypeError:
            return self.model.encode(texts, convert_to_numpy=True).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        raise RuntimeError("BenchmarkProfile_A does not support text generation.")

    def can_synthesize(self) -> bool:
        return False


    @property
    def embedding_dimension(self) -> int:
        return 768


class BenchmarkProfile_B(InferenceProfile):
    def __init__(self):
        self.llm = None

    def load_models(self):
        from llama_cpp import Llama
        from router import HardwareProfiler
        model_path = os.path.join(os.getcwd(), "model_2", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
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
            "embedding": True, # Optimize: single instance for both embed and chat
        }
        self.llm = Llama(**common_kwargs)

    def get_embedding(self, text: str) -> List[float]:
        result = self.llm.create_embedding(text)
        emb_data = result['data'][0]['embedding']
        if isinstance(emb_data, list) and len(emb_data) > 0 and isinstance(emb_data[0], list):
            import numpy as np
            return np.mean(emb_data, axis=0).tolist()
        return emb_data

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self.llm.create_embedding(texts)
            embeddings = []
            for item in result.get("data", []):
                emb_data = item["embedding"]
                if isinstance(emb_data, list) and emb_data and isinstance(emb_data[0], list):
                    import numpy as np
                    embeddings.append(np.mean(emb_data, axis=0).tolist())
                else:
                    embeddings.append(emb_data)
            if len(embeddings) == len(texts):
                return embeddings
        except Exception:
            pass
        return super().get_embeddings(texts)

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        import json
        messages = [{"role": "user", "content": prompt}]
        
        # If schema is provided or JSON is requested, use our custom streaming cutoff logic
        # rather than llama_cpp's strict grammar which causes infinite loops.
        is_json_request = schema is not None or "json" in prompt.lower()
        
        if is_json_request:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                stop=["<|im_end|>"]
            )
            buffer = ""
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content = delta["content"]
                    buffer += content
                    
                    if "}" in content or "]" in content:
                        try:
                            if "{" in buffer and "}" in buffer:
                                json_str = buffer[buffer.find("{"):buffer.rfind("}")+1]
                                # Test parse
                                json.loads(json_str)
                                # If it succeeds, cut the generation!
                                return json_str
                        except json.JSONDecodeError:
                            pass
                            
            # Fallback if loop finishes without valid JSON
            try:
                if "{" in buffer and "}" in buffer:
                    json_str = buffer[buffer.find("{"):buffer.rfind("}")+1]
                    return json_str
            except json.JSONDecodeError:
                pass
            return buffer
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens
            )
            return response["choices"][0]["message"]["content"]

    def can_synthesize(self) -> bool:
        return True

    @property
    def embedding_dimension(self) -> int:
        return 896


class BenchmarkProfile_C(InferenceProfile):
    def __init__(self):
        self.embedder = None
        self.llm = None

    def load_models(self):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        from router import HardwareProfiler
        
        model_path_1 = os.path.join(os.getcwd(), "model_1")
        if not os.path.exists(model_path_1):
            model_path_1 = "jinaai/jina-embeddings-v2-small-en"
        device = HardwareProfiler.get_optimal_device()
        self.embedder = SentenceTransformer(model_path_1, device=device, trust_remote_code=True)
        
        model_path_2 = os.path.join(os.getcwd(), "model_2", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
        if not os.path.exists(model_path_2):
            raise FileNotFoundError(f"Model not found at {model_path_2}")
        gpu_layers = -1 if device == "cuda" else 0
        self.llm = Llama(model_path=model_path_2, n_ctx=4096, mmap=True, verbose=False, n_gpu_layers=gpu_layers)

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedder.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except TypeError:
            return self.embedder.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.embedder.encode(texts, convert_to_numpy=True, task="retrieval").tolist()
        except TypeError:
            return self.embedder.encode(texts, convert_to_numpy=True).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_format = None
        if schema:
            response_format = {"type": "json_object", "schema": schema}
        elif "json" in prompt.lower():
            response_format = {"type": "json_object"}
            
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format
        )
        return response["choices"][0]["message"]["content"]

    def can_synthesize(self) -> bool:
        return True

    @property
    def embedding_dimension(self) -> int:
        return 768


class ModelManager:
    _instance = None
    # BUG 14 FIX: Lock for thread-safe singleton creation.
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        # BUG 14 FIX: Double-checked locking pattern to prevent two threads from
        # simultaneously creating separate ModelManager instances under concurrent load.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelManager()
        return cls._instance

    def __init__(self):
        self.profile = None
        # Benchmarking is OFF by default. It must be explicitly enabled via the
        # /api/benchmark/toggle endpoint so it doesn't flood the log during normal
        # operation (e.g. LocalRAG.build_index embedding calls on startup).
        self.benchmarking_enabled = False
        self._init_lock = threading.Lock()

    def initialize_profile(self, profile_type: str):
        with self._init_lock:
            previous_profile = self.profile
            self.profile = None
            if profile_type == "A":
                new_profile = BenchmarkProfile_A()
            elif profile_type == "B":
                new_profile = BenchmarkProfile_B()
            elif profile_type == "C":
                new_profile = BenchmarkProfile_C()
            else:
                self.profile = previous_profile
                raise ValueError(f"Unknown profile type: {profile_type}")
                
            logging.info(f"Loading BenchmarkProfile_{profile_type} models persistently...")
            try:
                new_profile.load_models()
            except Exception:
                self.profile = previous_profile
                raise
            else:
                self.profile = new_profile
                self._log_overhead(f"Initialized Profile_{profile_type}")

    def _log_overhead(self, context: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        bench_logger.info(f"[{context}] System RAM Usage: {mem_info.rss / (1024**2):.2f} MB")

    def get_embedding(self, text: str) -> List[float]:
        if not self.profile:
            raise RuntimeError("ModelManager profile not initialized.")
        start = time.perf_counter()
        result = self.profile.get_embedding(text)
        latency = time.perf_counter() - start
        if self.benchmarking_enabled:
            bench_logger.info(f"[Embedding] Latency: {latency:.4f}s")
        return result

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.profile:
            raise RuntimeError("ModelManager profile not initialized.")
        start = time.perf_counter()
        result = self.profile.get_embeddings(texts)
        latency = time.perf_counter() - start
        if self.benchmarking_enabled:
            bench_logger.info(f"[EmbeddingBatch] Count: {len(texts)} Latency: {latency:.4f}s")
        return result

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:
        if not self.profile:
            raise RuntimeError("ModelManager profile not initialized.")
        start = time.perf_counter()
        result = self.profile.generate_text(prompt, max_tokens, schema)
        latency = time.perf_counter() - start
        if self.benchmarking_enabled:
            bench_logger.info(f"[Generation] Latency: {latency:.4f}s")
            self._log_overhead("Post-Generation")
        return result

    def can_synthesize(self) -> bool:
        if not self.profile:
            return False
        return self.profile.can_synthesize()

    @property
    def embedding_dimension(self) -> int:
        if not self.profile:
            return 768
        return self.profile.embedding_dimension
