import os
import time
import threading
import psutil
import logging
from abc import ABC, abstractmethod
from typing import List

import sys
import types
from config import MODELS_DIR, LOGS_DIR

# Robust package mock for transformers.onnx for jinaai / HuggingFace model compatibility
class DummyPackage(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
    def __getattr__(self, attr):
        if attr in ('__file__', '__spec__'):
            raise AttributeError(attr)
        val = type(attr, (object,), {})
        setattr(self, attr, val)
        return val

if 'transformers.onnx' not in sys.modules:
    sys.modules['transformers.onnx'] = DummyPackage('transformers.onnx')
if 'transformers.onnx.utils' not in sys.modules:
    sys.modules['transformers.onnx.utils'] = DummyPackage('transformers.onnx.utils')

# Defensive fallback for timm.data compatibility with newer/older versions
try:
    import timm.data
    if not hasattr(timm.data, 'ImageNetInfo'):
        timm.data.ImageNetInfo = type('ImageNetInfo', (object,), {})
    if not hasattr(timm.data, 'infer_imagenet_subset'):
        timm.data.infer_imagenet_subset = lambda *args, **kwargs: None
except ImportError:
    pass

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

# Defensive monkeypatch for Qwen3Model __init__ dtype parameter compatibility
try:
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    _orig_qwen3_init = qwen3_mod.Qwen3Model.__init__
    def _patched_qwen3_init(self, config, *args, **kwargs):
        kwargs.pop('dtype', None)
        return _orig_qwen3_init(self, config, *args, **kwargs)
    qwen3_mod.Qwen3Model.__init__ = _patched_qwen3_init
except Exception:
    pass

class GGUFEmbeddingModel:
    def __init__(self, gguf_path: str, device: str = None):
        from llama_cpp import Llama
        from router import HardwareProfiler
        if not device:
            device = HardwareProfiler.get_embedder_device()
        gpu_layers = -1 if device == "cuda" else 0
        n_threads = max(1, (os.cpu_count() or 4) // 2)
        self.llama = Llama(model_path=gguf_path, embedding=True, verbose=False, n_gpu_layers=gpu_layers, n_threads=n_threads, n_ctx=4096)
        res = self.llama.create_embedding("test")
        emb = res['data'][0]['embedding']
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            self._dim = len(emb[0])
        else:
            self._dim = len(emb)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        chunk_size = 64
        for i in range(0, len(texts), chunk_size):
            chunk = [(t.strip() or " ")[:4000] for t in texts[i:i + chunk_size]]
            try:
                res = self.llama.create_embedding(chunk)
                for item in res['data']:
                    emb = item['embedding']
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                        seq_len = len(emb)
                        h_size = len(emb[0])
                        emb = [sum(emb[k][j] for k in range(seq_len)) / seq_len for j in range(h_size)]
                    embeddings.append(emb)
            except Exception:
                for t_clean in chunk:
                    try:
                        res = self.llama.create_embedding(t_clean)
                        emb = res['data'][0]['embedding']
                        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                            seq_len = len(emb)
                            h_size = len(emb[0])
                            emb = [sum(emb[k][j] for k in range(seq_len)) / seq_len for j in range(h_size)]
                        embeddings.append(emb)
                    except Exception:
                        embeddings.append([0.0] * self._dim)
        if convert_to_numpy:
            return np.array(embeddings, dtype=np.float32)
        return embeddings

def _load_embedder_model(emb_path: str, emb_device: str):
    if os.path.exists(emb_path):
        if os.path.isfile(emb_path) and emb_path.lower().endswith(".gguf"):
            return GGUFEmbeddingModel(emb_path, device=emb_device)
        if os.path.isdir(emb_path):
            ggufs = [f for f in os.listdir(emb_path) if f.lower().endswith(".gguf")]
            if ggufs:
                return GGUFEmbeddingModel(os.path.join(emb_path, ggufs[0]), device=emb_device)

    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from sentence_transformers import SentenceTransformer
    if emb_device == "mps":
        emb_device = "cpu"
    return SentenceTransformer(
        emb_path, 
        device=emb_device, 
        trust_remote_code=True, 
        model_kwargs={'low_cpu_mem_usage': False}
    )

# Setup benchmarking logger
bench_logger = logging.getLogger("Benchmark")
bench_logger.setLevel(logging.INFO)
if not bench_logger.handlers:
    os.makedirs(LOGS_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "benchmark_metrics.log"), encoding="utf-8")
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
    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:
        pass

    @abstractmethod
    def can_synthesize(self) -> bool:
        pass

    @abstractmethod
    def can_feedback_check(self) -> bool:
        pass

    def has_translator_pass(self) -> bool:
        return False

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
        
        if not embedder_name or embedder_name == "auto":
            emb_dir = os.path.join(MODELS_DIR, "embeddings")
            available = sorted([d for d in os.listdir(emb_dir) if os.path.isdir(os.path.join(emb_dir, d))], key=lambda x: x.lower()) if os.path.exists(emb_dir) else []
            if not available:
                raise ValueError("Profile A requires an embedder_name and no local embeddings found")
            embedder_name = available[0]
            
        self.embedder_name = embedder_name
        self.llm_name = None
        
        model_path = os.path.join(MODELS_DIR, "embeddings", embedder_name)
        if not os.path.exists(model_path):
            model_path = embedder_name  # Fallback to HuggingFace hub if local missing
            
        device = HardwareProfiler.get_embedder_device()
        self.model = _load_embedder_model(model_path, device)

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except TypeError:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch encode all texts in one vectorised call — O(N) not O(N²)."""
        if not texts:
            return []
        try:
            return self.model.encode(texts, convert_to_numpy=True, task="retrieval", batch_size=64).tolist()
        except TypeError:
            return self.model.encode(texts, convert_to_numpy=True, batch_size=64).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:
        raise RuntimeError("Profile A (Embedding Only) does not support text generation.")

    def can_synthesize(self) -> bool:
        return False
        
    def can_feedback_check(self) -> bool:
        return False
        
    def feedback_check(self, generated_code: str) -> str:
        return generated_code  # No feedback loop in Profile A

    @property
    def embedding_dimension(self) -> int:
        # Dynamically read from the loaded model to support any embedder (B-13 fix)
        if self.model is not None:
            try:
                return self.model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return 768  # Fallback only


class BenchmarkProfile_B(InferenceProfile):
    def __init__(self):
        self.llm = None
        self._dim = 896

    def load_models(self, embedder_name: str, llm_name: str):
        from llama_cpp import Llama
        from router import HardwareProfiler
        
        if not llm_name or llm_name == "auto":
            llm_base_dir = os.path.join(MODELS_DIR, "llms")
            available_llms = sorted([d for d in os.listdir(llm_base_dir) if os.path.isdir(os.path.join(llm_base_dir, d))], key=lambda x: x.lower()) if os.path.exists(llm_base_dir) else []
            if not available_llms:
                raise ValueError("Profile B requires an llm_name and no local LLMs found")
            llm_name = available_llms[0]
            
        # Profile B embeds through its selected LLM. Include that model in the
        # RAG cache identity so vectors are never reused across GGUFs.
        self.embedder_name = llm_name
        self.llm_name = llm_name
            
        model_dir = os.path.join(MODELS_DIR, "llms", llm_name)
        # Find the .gguf file inside the directory
        all_ggufs = [f for f in os.listdir(model_dir) if f.lower().endswith('.gguf')] if os.path.exists(model_dir) else []
        gguf_files = sorted(all_ggufs, key=lambda f: (1 if '-of-' in f.lower() else 0, f))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf model found in {model_dir}")
        model_path = os.path.join(model_dir, gguf_files[0])

        device = HardwareProfiler.get_llm_device()
        gpu_layers = -1 if device == "cuda" else 0
        # Cap threads at physical core count to prevent hyperthreading contention
        n_threads = max(1, (os.cpu_count() or 4) // 2)
        common_kwargs = {
            "model_path": model_path,
            "n_ctx": 4096,  # Must be large enough for Planner generation prompts
            "mmap": True,
            "verbose": False,
            "n_gpu_layers": gpu_layers,
            "embedding": True,
            "n_threads": n_threads,
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
        if not text.strip():
            text = " "
        # Truncate to roughly 4000 chars to avoid exceeding 4096 tokens
        if len(text) > 4000:
            text = text[:4000]
        try:
            result = self.llm.create_embedding(text)
            emb = result['data'][0]['embedding']
            
            # Mean pool if seq-level
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                seq_len = len(emb)
                hidden_size = len(emb[0])
                emb = [sum(emb[i][j] for i in range(seq_len)) / seq_len for j in range(hidden_size)]
                
            if not emb:
                return [0.0] * self._dim
                
            # Auto-learn dim on first successful call if we were wrong
            if len(emb) > 0 and self._dim != len(emb):
                self._dim = len(emb)
                
            return [float(x) for x in emb]
        except Exception as e:
            bench_logger.error(f"Error creating embedding for text: {e}")
            return [0.0] * self._dim

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding via llama_cpp native batch API with chunking and post-processing 
        to ensure homogeneous array lengths. Chunking avoids hitting the n_ctx limit."""
        if not texts:
            return []
            
        cleaned = [(t.strip() or " ")[:4000] for t in texts]
        results: List[List[float]] = []
        
        # Batch in chunks of 128 to improve GPU utilization
        chunk_size = 128
        total_chunks = (len(cleaned) + chunk_size - 1) // chunk_size
        for i in range(0, len(cleaned), chunk_size):
            chunk = cleaned[i:i + chunk_size]
            chunk_idx = (i // chunk_size) + 1
            if total_chunks > 1:
                print(f"    -> Embedding batch {chunk_idx}/{total_chunks}...")
            try:
                raw_batch = self.llm.create_embedding(chunk)
                embeddings = [item['embedding'] for item in raw_batch['data']]
                
                for emb in embeddings:
                    # If llama_cpp returns [seq_len, hidden_size] (a list of lists), perform mean pooling
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                        seq_len = len(emb)
                        hidden_size = len(emb[0])
                        pooled = [sum(emb[i][j] for i in range(seq_len)) / seq_len for j in range(hidden_size)]
                        emb = pooled
                    
                    # Coerce to list of floats
                    if not isinstance(emb, list):
                        try:
                            emb = list(emb)
                        except Exception:
                            emb = []
                            
                    # Auto-learn dim on first successful call
                    if len(emb) > 0 and self._dim != len(emb) and not getattr(self, '_dim_locked', False):
                        self._dim = len(emb)
                        self._dim_locked = True
                    
                    # Force exact dimension
                    if len(emb) != self._dim:
                        if len(emb) < self._dim:
                            emb = emb + [0.0] * (self._dim - len(emb))
                        else:
                            emb = emb[:self._dim]
                            
                    results.append([float(x) for x in emb])
            except Exception as e:
                bench_logger.error(f"Batch embedding chunk failed (chunk size {len(chunk)}): {e}", exc_info=True)
                # Re-raise instead of zero-filling to prevent poisoning the FAISS index
                raise RuntimeError(f"Embedding batch failed: {e}. Refusing to fill with zero vectors.") from e
                
        # Ensure ALL results are exactly self._dim (fix for inhomogeneous shapes if dim was learned mid-batch)
        for i in range(len(results)):
            if len(results[i]) != self._dim:
                if len(results[i]) < self._dim:
                    results[i] = results[i] + [0.0] * (self._dim - len(results[i]))
                else:
                    results[i] = results[i][:self._dim]
            
        return results


    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stop": ["```\n", "<|im_end|>"],
            "temperature": 0.1,
            "top_p": 0.95
        }
        if schema:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema
            }
        
        try:
            response = self.llm.create_chat_completion(**kwargs)
        except Exception as e:
            bench_logger.error(f"LLM chat completion failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM inference failed: {e}") from e
        return response['choices'][0]['message']['content'].strip()

    def can_synthesize(self) -> bool:
        return True
        
    def can_feedback_check(self) -> bool:
        return True
        
    def feedback_check(self, generated_code: str) -> str:
        prompt = f"Rewrite this Python code to use clean, standard variable names (like df for dataframes). Do not change what the code does, only rename the variables to be professional. Return ONLY the code inside ```python block.\n\n```python\n{generated_code.strip()}\n```"
        sys_prompt = "You are a professional Python engineer. Your task is to clean up variable names in the provided code."
        corrected = self.generate_text(prompt, max_tokens=1024, system_prompt=sys_prompt)
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
        
        if not embedder_name or embedder_name == "auto":
            emb_dir = os.path.join(MODELS_DIR, "embeddings")
            available = sorted([d for d in os.listdir(emb_dir) if os.path.isdir(os.path.join(emb_dir, d))], key=lambda x: x.lower()) if os.path.exists(emb_dir) else []
            if not available:
                raise ValueError("Profile C requires an embedder_name and no local embeddings found")
            embedder_name = available[0]
            
        if not llm_name or llm_name == "auto":
            llm_base_dir = os.path.join(MODELS_DIR, "llms")
            available_llms = sorted([d for d in os.listdir(llm_base_dir) if os.path.isdir(os.path.join(llm_base_dir, d))], key=lambda x: x.lower()) if os.path.exists(llm_base_dir) else []
            if not available_llms:
                raise ValueError("Profile C requires an llm_name and no local LLMs found")
            llm_name = available_llms[0]
             
        self.embedder_name = embedder_name
        self.llm_name = llm_name
             
        # Load Embedder
        emb_path = os.path.join(MODELS_DIR, "embeddings", embedder_name)
        if not os.path.exists(emb_path):
            emb_path = embedder_name
            
        llm_device = HardwareProfiler.get_llm_device()
        # When an LLM is active on GPU, route the text embedder to CPU to preserve 100% CUDA VRAM for the 7B LLM
        emb_device = "cpu" if llm_device == "cuda" else HardwareProfiler.get_embedder_device()
        self.embedder = _load_embedder_model(emb_path, emb_device)
        detected_dimension = getattr(self.embedder, "get_sentence_embedding_dimension", lambda: None)()
        if not detected_dimension:
            try:
                sample = self.get_embedding("test")
                detected_dimension = len(sample)
            except Exception:
                pass
        if detected_dimension:
            self._dim = int(detected_dimension)
        
        # Load LLM
        llm_dir = os.path.join(MODELS_DIR, "llms", llm_name)
        all_ggufs = [f for f in os.listdir(llm_dir) if f.lower().endswith('.gguf')] if os.path.exists(llm_dir) else []
        gguf_files = sorted(all_ggufs, key=lambda f: (1 if '-of-' in f.lower() else 0, f))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf model found in {llm_dir}")
        llm_path = os.path.join(llm_dir, gguf_files[0])
        
        # Memory cleanup before initializing heavy LLM
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gpu_layers = -1 if llm_device == "cuda" else 0
        n_threads = max(1, (os.cpu_count() or 4) // 2)

        # Attempt full GPU offloading; fallback to reduced context / partial offloading if VRAM is tight
        llm_obj = None
        attempt_params = [
            {"n_gpu_layers": gpu_layers, "n_ctx": 4096},
            {"n_gpu_layers": gpu_layers, "n_ctx": 2048},
            {"n_gpu_layers": 20, "n_ctx": 2048},
            {"n_gpu_layers": 0, "n_ctx": 2048},
        ]
        
        last_err = None
        for params in attempt_params:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                llm_obj = Llama(
                    model_path=llm_path,
                    n_ctx=params["n_ctx"],
                    mmap=True,
                    verbose=False,
                    n_gpu_layers=params["n_gpu_layers"],
                    embedding=False,
                    n_threads=n_threads,
                )
                break
            except Exception as e:
                last_err = e
                logging.warning(f"Llama load failed with params {params}: {e}. Retrying with fallback...")

        if llm_obj is None:
            raise RuntimeError(f"Failed to initialize Llama model from {llm_path}: {last_err}")
        self.llm = llm_obj

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedder.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except Exception:
            return self.embedder.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Vectorised batch encode — chunked GPU/CPU calls for all N texts."""
        if not texts:
            return []
        try:
            return self.embedder.encode(texts, convert_to_numpy=True, task="retrieval", batch_size=64, show_progress_bar=False).tolist()
        except Exception:
            return self.embedder.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stop": ["<|im_end|>"],
            "temperature": 0.1,
            "top_p": 0.95
        }
        if schema:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema
            }
        
        response = self.llm.create_chat_completion(**kwargs)
        return response['choices'][0]['message']['content'].strip()

    def can_synthesize(self) -> bool:
        return True
        
    def can_feedback_check(self) -> bool:
        return True
        
    def feedback_check(self, generated_code: str) -> str:
        import ast
        import re
        prompt = f"Rewrite this Python code to use clean, standard variable names (like df for dataframes). Do not change what the code does or remove any function arguments, only rename the variables to be professional. Return ONLY the code inside ```python block.\n\n```python\n{generated_code.strip()}\n```"
        sys_prompt = "You are a professional Python engineer. Your task is to clean up variable names in the provided code without removing any function arguments or literal strings."
        try:
            corrected = self.generate_text(prompt, max_tokens=1024, system_prompt=sys_prompt)
            if "```python" in corrected:
                corrected = corrected.split("```python")[1].split("```")[0].strip()
            elif "```" in corrected:
                corrected = corrected.split("```")[1].split("```")[0].strip()
            
            # AST Parity Check: Verify literal string parameters were not stripped by LLM rewrite
            gen_literals = set(re.findall(r"['\"]([^'\"]+)['\"]", generated_code))
            cor_literals = set(re.findall(r"['\"]([^'\"]+)['\"]", corrected))
            if not gen_literals.issubset(cor_literals):
                return generated_code

            tree = ast.parse(corrected)
            return corrected if corrected.strip() else generated_code
        except Exception:
            return generated_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim

class BenchmarkProfile_D(BenchmarkProfile_C):
    def can_synthesize(self) -> bool:
        return False
        
    def can_feedback_check(self) -> bool:
        return True

class BenchmarkProfile_E(BenchmarkProfile_C):
    """Profile E: Includes an initial LLM Translation pre-pass before standard planning & execution."""
    def has_translator_pass(self) -> bool:
        return True

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
                new_profile.load_models(embedder_name, llm_name)
            elif profile_type == "B":
                new_profile = BenchmarkProfile_B()
                new_profile.load_models(embedder_name, llm_name)
            elif profile_type == "C":
                new_profile = BenchmarkProfile_C()
                new_profile.load_models(embedder_name, llm_name)
            elif profile_type == "D":
                new_profile = BenchmarkProfile_D()
                new_profile.load_models(embedder_name, llm_name)
            elif profile_type == "E":
                new_profile = BenchmarkProfile_E()
                new_profile.load_models(embedder_name, llm_name)
            else:
                raise ValueError(f"Unknown profile: {profile_type}")
            
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

    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:
        if self.active_profile is None: return ""
        if not self.active_profile.can_synthesize():
            logging.info("Active profile has no text generator; using deterministic planner fallback.")
            return ""
        t0 = time.time()
        # B-4 fix: forward system_prompt so the LLM actually uses its instructions
        text = self.active_profile.generate_text(prompt, max_tokens, schema, system_prompt=system_prompt)
        t1 = time.time()
        if self.benchmarking_enabled:
            bench_logger.info(f"[Generation] Latency: {t1-t0:.4f}s")
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            bench_logger.info(f"[Post-Generation] System RAM Usage: {mem:.2f} MB")
        return text

    def can_synthesize(self) -> bool:
        if self.active_profile is None: return False
        return self.active_profile.can_synthesize()
        
    def can_feedback_check(self) -> bool:
        if self.active_profile is None: return False
        return self.active_profile.can_feedback_check()

    def has_translator_pass(self) -> bool:
        if self.active_profile is None: return False
        return self.active_profile.has_translator_pass()
        
    def feedback_check(self, generated_code: str) -> str:
        if self.active_profile is None: return generated_code
        return self.active_profile.feedback_check(generated_code)

    @property
    def embedding_dimension(self) -> int:
        if self.active_profile is None: return 0
        return self.active_profile.embedding_dimension
