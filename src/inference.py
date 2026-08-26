"""
src/inference.py - Neuro-Symbolic Topological Lattice (NSTL)
Multi-Profile Inference Engine with Aggressive VRAM Recycling.
"""

from __future__ import annotations
import gc
import os
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from log_config import get_logger
from config import MODELS_DIR

logger = get_logger("inference")


class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self, embedder_name: str, llm_name: str):
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(t) for t in texts]

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
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
    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        pass


class BenchmarkProfile_A(InferenceProfile):
    """Profile A: Embedding Only."""
    def __init__(self):
        self.model = None
        self._dim = 384
        self.embedder_name = "default"

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from router import HardwareProfiler

        self.embedder_name = embedder_name or "auto"
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        if not os.path.exists(emb_path):
            emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
            if os.path.exists(emb_base_dir):
                available = sorted([
                    d for d in os.listdir(emb_base_dir)
                    if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
                ])
                if available:
                    self.embedder_name = available[0]
                    emb_path = os.path.join(emb_base_dir, self.embedder_name)

        device = HardwareProfiler.get_optimal_device()
        try:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs={"default_task": "retrieval"})
        except Exception:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True)

        self._dim = self.model.get_sentence_embedding_dimension() or 384
        logger.info(f"[PROFILE A] Loaded embedder '{self.embedder_name}' (dim={self._dim}) on {device.upper()}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except Exception:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return self.model.encode(texts, convert_to_numpy=True, task="retrieval", batch_size=128).tolist()
        except Exception:
            return self.model.encode(texts, convert_to_numpy=True, batch_size=128).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile A does not support text generation.")

    def can_synthesize(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_C(InferenceProfile):
    """Profile C: Dedicated Embedder + GGUF LLM."""
    def __init__(self):
        self.embedder = None
        self.llm = None
        self._dim = 384
        self.embedder_name = "default"
        self.llm_name = "default"

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        from router import HardwareProfiler

        device = HardwareProfiler.get_optimal_device()

        # 1. Load Embedder
        self.embedder_name = embedder_name or "auto"
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        if not os.path.exists(emb_path):
            emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
            if os.path.exists(emb_base_dir):
                available = sorted([
                    d for d in os.listdir(emb_base_dir)
                    if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
                ])
                if available:
                    self.embedder_name = available[0]
                    emb_path = os.path.join(emb_base_dir, self.embedder_name)

        try:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs={"default_task": "retrieval"})
        except Exception:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True)

        self._dim = self.embedder.get_sentence_embedding_dimension() or 384

        # 2. Load LLM
        self.llm_name = llm_name or "auto"
        llm_dir = os.path.join(MODELS_DIR, "llms", self.llm_name)
        if not os.path.exists(llm_dir):
            llm_base_dir = os.path.join(MODELS_DIR, "llms")
            if os.path.exists(llm_base_dir):
                available_llms = sorted([d for d in os.listdir(llm_base_dir) if os.path.isdir(os.path.join(llm_base_dir, d))])
                if available_llms:
                    self.llm_name = available_llms[0]
                    llm_dir = os.path.join(llm_base_dir, self.llm_name)

        ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")] if os.path.exists(llm_dir) else []
        if not ggufs:
            raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
        model_file = os.path.join(llm_dir, ggufs[0])

        gpu_layers = -1 if device == "cuda" else 0
        # Bound context to 2048 to prevent VRAM bloat
        self.llm = Llama(model_path=model_file, n_ctx=2048, n_gpu_layers=gpu_layers, verbose=False)
        logger.info(f"[PROFILE C] Loaded Embedder '{self.embedder_name}' + LLM '{self.llm_name}' on {device.upper()}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedder.encode([text], convert_to_numpy=True, task="retrieval")[0].tolist()
        except Exception:
            return self.embedder.encode([text], convert_to_numpy=True)[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return self.embedder.encode(texts, convert_to_numpy=True, task="retrieval", batch_size=128).tolist()
        except Exception:
            return self.embedder.encode(texts, convert_to_numpy=True, batch_size=128).tolist()

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.1, "top_p": 0.95}
        if schema:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}

        response = self.llm.create_chat_completion(**kwargs)
        return response['choices'][0]['message']['content'].strip()

    def can_synthesize(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        prompt = f"Fix the runtime error in this code and return ONLY the corrected code inside ```python ```:\n\nERROR:\n{traceback_error}\n\nCODE:\n```python\n{failing_code}\n```"
        try:
            return self.generate_text(prompt, max_tokens=1024)
        except Exception as e:
            logger.error(f"[FEEDBACK CHECK ERROR] {e}")
            return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_D(BenchmarkProfile_C):
    def can_synthesize(self) -> bool:
        return False


class BenchmarkProfile_E(BenchmarkProfile_C):
    def has_translator_pass(self) -> bool:
        return True


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.active_profile: Optional[InferenceProfile] = None
        self.current_profile_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> ModelManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def profile(self) -> Optional[InferenceProfile]:
        return self.active_profile

    def initialize_profile(self, profile_type: str, embedder_name: str = "", llm_name: str = ""):
        with self._lock:
            # 1. Cleanly tear down existing models to free GPU VRAM & CPU RAM
            if self.active_profile is not None:
                if hasattr(self.active_profile, 'llm') and self.active_profile.llm is not None:
                    try:
                        self.active_profile.llm.close()
                    except Exception:
                        pass
                    self.active_profile.llm = None
                if hasattr(self.active_profile, 'embedder') and self.active_profile.embedder is not None:
                    self.active_profile.embedder = None
                if hasattr(self.active_profile, 'model') and self.active_profile.model is not None:
                    self.active_profile.model = None
                self.active_profile = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # 2. Instantiate new profile
            p_type = profile_type.upper()
            if p_type == "A":
                prof = BenchmarkProfile_A()
            elif p_type in ("C", "B"):
                prof = BenchmarkProfile_C()
            elif p_type == "D":
                prof = BenchmarkProfile_D()
            elif p_type == "E":
                prof = BenchmarkProfile_E()
            else:
                raise ValueError(f"Unknown profile type: {profile_type}")

            prof.load_models(embedder_name, llm_name)
            self.active_profile = prof
            self.current_profile_name = p_type

    def get_embedding(self, text: str) -> List[float]:
        return self.active_profile.get_embedding(text) if self.active_profile else []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.active_profile.get_embeddings(texts) if self.active_profile else []

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        if not self.active_profile:
            return ""
        return self.active_profile.generate_text(prompt, max_tokens, schema, system_prompt=system_prompt)

    def can_synthesize(self) -> bool:
        return self.active_profile.can_synthesize() if self.active_profile else False

    def can_feedback_check(self) -> bool:
        return self.active_profile.can_feedback_check() if self.active_profile else False

    def has_translator_pass(self) -> bool:
        return self.active_profile.has_translator_pass() if self.active_profile else False

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return self.active_profile.feedback_check(failing_code, traceback_error) if self.active_profile else failing_code

    @property
    def embedding_dimension(self) -> int:
        return self.active_profile.embedding_dimension if self.active_profile else 384
