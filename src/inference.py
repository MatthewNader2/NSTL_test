"""
src/inference.py - Neuro-Symbolic Topological Lattice (NSTL)
Multi-Profile Inference Engine with Dynamic Hardware Introspection,
Agnostic Semantic Compilation, and Thread-Safe Generation.
"""

from __future__ import annotations
import gc
import json
import os
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

try:
    from utils import is_split_gguf_shard
except ImportError:
    from .utils import is_split_gguf_shard

import numpy as np

# Torch is required only by neural inference profiles; the symbolic toolchain
# (CLI compile/validate/harvest) runs without it.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from log_config import get_logger
except ImportError:
    try:
        from .log_config import get_logger
    except ImportError:
        import logging
        get_logger = logging.getLogger

try:
    from .config import MODELS_DIR, settings
    from .utils import extract_code_from_llm_response
except (ImportError, ValueError):
    from config import MODELS_DIR, settings
    from utils import extract_code_from_llm_response

# Configure PyTorch CUDA memory allocator to prevent memory fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger("inference")


def resolve_embedding_dimension(model: Any, fallback: int = 384) -> int:
    """
    Resolves the embedding dimension of a loaded SentenceTransformer model
    across library versions.
    """
    modern = getattr(model, "get_embedding_dimension", None)
    if callable(modern):
        try:
            dim = int(modern())
            if dim > 0:
                return dim
        except Exception:
            pass

    legacy = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(legacy):
        try:
            dim = int(legacy())
            if dim > 0:
                return dim
        except TypeError:
            name = getattr(model, "_model_name", None) or getattr(model, "model_name_or_path", None)
            if isinstance(name, str) and name:
                try:
                    dim = int(legacy(name))
                    if dim > 0:
                        return dim
                except Exception:
                    pass
        except Exception:
            pass

    try:
        probe = model.encode(["nstl dimension probe"], convert_to_numpy=True)
        dim = int(probe.shape[-1])
        if dim > 0:
            return dim
    except Exception:
        pass

    logger.warning(f"[INFERENCE] Could not resolve embedding dimension; using fallback {fallback}.")
    return fallback


def get_adaptive_batch_size(device: str = "cuda", min_batch: int = 8, max_batch: int = 128) -> int:
    """
    Computes an optimal batch size dynamically based on available device memory headroom.
    Fully hardware-agnostic: tracks continuous free VRAM headroom without hardcoded card thresholds.
    """
    if not TORCH_AVAILABLE or "cuda" not in str(device).lower() or not torch.cuda.is_available():
        return 32

    try:
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        # Continuous logarithmic scaling mapped to powers of 2 (approx. 8 items per 1 GB of free headroom)
        power = int(np.clip(np.floor(np.log2(max(1.0, free_gb * 8))), 3, 7))
        calculated_bs = int(2 ** power)
        return int(max(min_batch, min(max_batch, calculated_bs)))
    except Exception as e:
        logger.debug(f"[HARDWARE TELEMETRY] Adaptive batch size fallback: {e}")
        return 32


class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self, embedder_name: str, llm_name: str):
        pass

    @abstractmethod
    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        pass

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        return [self.get_embedding(t, mode=mode) for t in texts]

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


def select_optimal_embedder(requested_name: str = "") -> str:
    """
    Selects the optimal local embedding model based on precomputed cache presence and footprint efficiency.
    Scoring is dynamically normalized across available candidate models without hardcoded architecture
    substrings, fixed cache naming prefixes, or arbitrary byte constants.
    """
    emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
    if not os.path.exists(emb_base_dir):
        return requested_name or "default"

    available = sorted([
        d for d in os.listdir(emb_base_dir)
        if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
    ])
    if not available:
        return requested_name or "default"

    req = (requested_name or "").strip()
    if req and req not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand
    else:
        # Default / auto selection prioritizes the nano embedder when present
        m_nano = next((m for m in available if "nano" in m.lower()), None)
        if m_nano is not None:
            return m_nano

    # Locate cache directories
    possible_cache_dirs = [
        os.path.join(os.path.dirname(MODELS_DIR), ".rag_cache"),
        os.path.join(MODELS_DIR, ".rag_cache"),
        os.path.abspath(".rag_cache")
    ]
    cache_dir = next((cd for cd in possible_cache_dirs if os.path.isdir(cd)), None)

    model_disk_sizes: Dict[str, int] = {}
    model_cache_sizes: Dict[str, int] = {}

    for cand in available:
        cand_path = os.path.join(emb_base_dir, cand)
        try:
            total_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(cand_path)
                for f in files
            )
        except Exception:
            total_size = 1
        model_disk_sizes[cand] = max(1, total_size)

        cand_cache_size = 0
        if cache_dir and os.path.isdir(cache_dir):
            cand_norm = cand.lower().replace("-", "_")
            try:
                for f in os.listdir(cache_dir):
                    f_lower = f.lower()
                    if (cand.lower() in f_lower or cand_norm in f_lower) and "cache" in f_lower:
                        cand_cache_size = max(cand_cache_size, os.path.getsize(os.path.join(cache_dir, f)))
            except Exception:
                pass
        model_cache_sizes[cand] = cand_cache_size

    min_disk = min(model_disk_sizes.values()) if model_disk_sizes else 1
    max_cache = max(model_cache_sizes.values()) if model_cache_sizes else 0

    best_model = available[0]
    best_score = -1.0

    for cand in available:
        efficiency = float(min_disk / model_disk_sizes[cand])
        coverage = float(model_cache_sizes[cand] / max_cache) if max_cache > 0 else 0.0

        score = 0.85 * coverage + 0.15 * efficiency
        logger.debug(f"[EMBEDDER EVAL] Model: {cand} | Coverage: {coverage:.2f} | Efficiency: {efficiency:.2f} | Score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_model = cand

    logger.info(f"[EMBEDDER SELECTION] Selected optimal model '{best_model}' (readiness score: {best_score:.3f})")
    return best_model


def select_optimal_llm(requested_name: str = "") -> str:
    """
    Selects the optimal local GGUF LLM model.
    Defaults to the 0.5b model when available, or matches the requested name.
    """
    llm_base_dir = os.path.join(MODELS_DIR, "llms")
    if not os.path.exists(llm_base_dir):
        return requested_name or "qwen2.5-coder-0.5b-instruct"

    available = sorted([
        d for d in os.listdir(llm_base_dir)
        if os.path.isdir(os.path.join(llm_base_dir, d)) and any(f.endswith(".gguf") for f in os.listdir(os.path.join(llm_base_dir, d)))
    ])
    if not available:
        return requested_name or "qwen2.5-coder-0.5b-instruct"

    req = (requested_name or "").strip()
    if req and req not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand

    # Default to 0.5b model when available
    m_05b = next((m for m in available if "0.5b" in m.lower()), None)
    if m_05b is not None:
        return m_05b

    return available[0]


def _encode_with_modes(
    model: Any,
    texts: List[str],
    mode: str = "query",
    batch_size: int = 32,
    lock: Optional[threading.Lock] = None
) -> List[List[float]]:
    """
    Robust embedding encoder applying model-specific prompt modes (query vs document)
    with adaptive fallback across SentenceTransformer models.
    Properly isolates CUDA OOM handling from non-OOM runtime exceptions.
    """
    if not texts:
        return []

    prompt_name = "query" if mode == "query" else "document"

    def _invoke(p_name: Optional[str], t_name: Optional[str], b_size: int, input_texts: List[str]):
        kwargs: Dict[str, Any] = {"convert_to_numpy": True, "batch_size": b_size}
        if p_name:
            kwargs["prompt_name"] = p_name
        if t_name:
            kwargs["task"] = t_name

        if lock is not None:
            with lock:
                if TORCH_AVAILABLE:
                    with torch.inference_mode():
                        return model.encode(input_texts, **kwargs)
                return model.encode(input_texts, **kwargs)
        else:
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    return model.encode(input_texts, **kwargs)
            return model.encode(input_texts, **kwargs)

    attempts = [
        (prompt_name, "retrieval"),
        (prompt_name, None),
        (None, "retrieval"),
        (None, None),
    ]

    last_sig_error = None

    for p_name, t_name in attempts:
        try:
            res = _invoke(p_name, t_name, batch_size, texts)
            return res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
        except (TypeError, ValueError) as arg_err:
            last_sig_error = arg_err
            continue
        except Exception as err:
            is_oom = TORCH_AVAILABLE and isinstance(err, torch.cuda.OutOfMemoryError)
            if not is_oom and "out of memory" not in str(err).lower():
                logger.error(f"[ENCODER] Fatal encoding error: {err}")
                raise err

            logger.warning(f"[ENCODER] CUDA OOM encountered at batch_size={batch_size}. Retrying with micro-batches.")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            micro_bs = max(1, batch_size // 4)
            micro_results: List[List[float]] = []
            try:
                for i in range(0, len(texts), micro_bs):
                    sub_batch = texts[i : i + micro_bs]
                    sub_res = _invoke(p_name, t_name, micro_bs, sub_batch)
                    if hasattr(sub_res, "tolist"):
                        micro_results.extend(sub_res.tolist())
                    else:
                        micro_results.extend([list(r) for r in sub_res])
                return micro_results
            except (TypeError, ValueError) as arg_err:
                last_sig_error = arg_err
                continue
            except Exception as sub_err:
                is_sub_oom = TORCH_AVAILABLE and isinstance(sub_err, torch.cuda.OutOfMemoryError)
                if is_sub_oom or "out of memory" in str(sub_err).lower():
                    logger.warning(f"[ENCODER] Micro-batch size {micro_bs} triggered OOM. Attempting next signature candidate.")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                logger.error(f"[ENCODER] Non-OOM error occurred in sub-batch loop: {sub_err}")
                raise sub_err

    # Final guarded fallback if model signature rejected prompt/task arguments
    try:
        if lock is not None:
            with lock:
                if TORCH_AVAILABLE:
                    with torch.inference_mode():
                        res = model.encode(texts, convert_to_numpy=True)
                else:
                    res = model.encode(texts, convert_to_numpy=True)
        else:
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    res = model.encode(texts, convert_to_numpy=True)
            else:
                res = model.encode(texts, convert_to_numpy=True)
        return res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
    except Exception as final_err:
        logger.error(f"[ENCODER] Fallback encoding failed: {final_err}")
        raise final_err from last_sig_error


class BenchmarkProfile_A(InferenceProfile):
    """Profile A: Embedding Only."""
    def __init__(self):
        self.model = None
        self._dim = 384
        self.embedder_name = "default"
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        self.embedder_name = select_optimal_embedder(embedder_name)
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        device = HardwareProfiler.get_optimal_device()
        is_jina = "jina" in self.embedder_name.lower()
        model_kwargs = {"default_task": "retrieval"} if is_jina else {}
        try:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        except Exception:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True)

        self._dim = resolve_embedding_dimension(self.model)
        logger.info(f"[PROFILE A] Loaded embedder '{self.embedder_name}' (dim={self._dim}) on {device.upper()}")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.model is None:
            return []
        device = getattr(self.model, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.model, texts, mode=mode, batch_size=batch_size, lock=self._lock)

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
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        device = HardwareProfiler.get_optimal_device()

        # 1. Load Embedder
        self.embedder_name = select_optimal_embedder(embedder_name)
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        is_jina = "jina" in self.embedder_name.lower()
        model_kwargs = {"default_task": "retrieval"} if is_jina else {}

        try:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        except Exception:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True)

        self._dim = resolve_embedding_dimension(self.embedder)

        # 2. Load LLM
        self.llm_name = select_optimal_llm(llm_name)
        llm_base_dir = os.path.join(MODELS_DIR, "llms")
        llm_dir = os.path.join(llm_base_dir, self.llm_name)
        if not os.path.isdir(llm_dir):
            raise FileNotFoundError(f"LLM directory not found: {llm_dir}")

        ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
        unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
        chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
        model_file = os.path.join(llm_dir, chosen_gguf)

        gpu_layers = -1 if device == "cuda" else 0
        self.llm = Llama(
            model_path=model_file,
            n_ctx=getattr(settings, "llm_context_length", 4096),
            n_gpu_layers=gpu_layers,
            verbose=False
        )
        logger.info(f"[PROFILE C] Loaded Embedder '{self.embedder_name}' + LLM '{self.llm_name}' on {device.upper()} (gpu_layers={gpu_layers})")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.embedder is None:
            return []
        device = getattr(self.embedder, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.embedder, texts, mode=mode, batch_size=batch_size, lock=self._lock)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": getattr(settings, "llm_temperature", 0.2),
            "top_p": getattr(settings, "llm_top_p", 0.95),
        }
        if schema:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}

        # Concurrency safety: synchronize queries to llama_cpp instance
        with self._lock:
            response = self.llm.create_chat_completion(**kwargs)
        return response['choices'][0]['message']['content'].strip()

    def can_synthesize(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        prompt = f"Fix the runtime error in this code and return ONLY the corrected code inside ```python ```:\n\nERROR:\n{traceback_error}\n\nCODE:\n```python\n{failing_code}\n```"
        try:
            raw = self.generate_text(prompt, max_tokens=1024)
            return extract_code_from_llm_response(raw)
        except Exception as e:
            logger.error(f"[FEEDBACK CHECK ERROR] {e}")
            return failing_code

    def has_semantic_compiler(self) -> bool:
        return self.llm is not None

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts structured semantic intent, parameters, and topological bindings using the LLM.
        Domain-agnostic with customizable schema and prompt contracts.
        """
        target_schema = schema or {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "domain": {"type": "string"},
                "inputs": {"type": "array", "items": {"type": "string"}},
                "outputs": {"type": "array", "items": {"type": "string"}},
                "operations": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object"},
                "slots": {"type": "object"},
                "effective_prompt": {"type": "string"},
            },
            "required": ["task", "domain", "operations"]
        }

        target_system_prompt = system_prompt or (
            "You are a structured semantic compiler for algorithmic pipelines and computational graphs. "
            "Given a user request, extract the execution task, domain, input entities, output artifacts, "
            "ordered sequence of operations, parameter constraints, and symbol slot bindings. "
            "Output ONLY valid JSON."
        )

        try:
            raw = self.generate_text(prompt, max_tokens=384, schema=target_schema, system_prompt=target_system_prompt)
            data = json.loads(raw)
            if isinstance(data, dict):
                # Ensure backward-compatible aliases for legacy consumers
                if "inputs" in data and "source_files" not in data:
                    data["source_files"] = data["inputs"]
                if "outputs" in data and "dest_files" not in data:
                    data["dest_files"] = data["outputs"]
                if "parameters" in data and "hyperparameters" not in data:
                    data["hyperparameters"] = data["parameters"]
                if "effective_prompt" not in data or not data["effective_prompt"]:
                    data["effective_prompt"] = prompt
                else:
                    try:
                        from utils import ensure_comprehensive_prompt
                        data["effective_prompt"] = ensure_comprehensive_prompt(prompt, data["effective_prompt"])
                    except Exception:
                        pass
                return data
        except Exception as e:
            logger.debug(f"[LLM] Semantic intent compilation fallback: {e}")

        return {
            "task": "generic",
            "domain": "generic",
            "inputs": [],
            "outputs": [],
            "source_files": [],
            "dest_files": [],
            "operations": [],
            "parameters": {},
            "hyperparameters": {},
            "slots": {},
            "effective_prompt": prompt,
        }

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_D(BenchmarkProfile_C):
    def can_synthesize(self) -> bool:
        return False


class BenchmarkProfile_E(BenchmarkProfile_C):
    def has_translator_pass(self) -> bool:
        return True


class BenchmarkProfile_S(BenchmarkProfile_C):
    """Profile S: Structured Semantic Compiler Profile."""
    pass


class ModelManager:
    _instance = None
    _lock = threading.RLock()

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
        with self._lock:
            return self.active_profile

    def cleanup(self):
        """Forces complete teardown of active models, releasing all GPU VRAM and CPU RAM."""
        with self._lock:
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
            self.current_profile_name = None

            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("[MODEL MANAGER] Completed VRAM cleanup and memory reclamation.")

    def initialize_profile(self, profile_type: str, embedder_name: str = "", llm_name: str = ""):
        with self._lock:
            self.cleanup()

            p_type = profile_type.upper()
            if p_type == "A":
                prof = BenchmarkProfile_A()
            elif p_type in ("C", "B"):
                prof = BenchmarkProfile_C()
            elif p_type == "D":
                prof = BenchmarkProfile_D()
            elif p_type == "E":
                prof = BenchmarkProfile_E()
            elif p_type == "S":
                prof = BenchmarkProfile_S()
            else:
                raise ValueError(f"Unknown profile type: {profile_type}")

            try:
                prof.load_models(embedder_name, llm_name)
                self.active_profile = prof
                self.current_profile_name = p_type
            except Exception as e:
                self.cleanup()
                raise e

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        with self._lock:
            return self.active_profile.get_embedding(text, mode=mode) if self.active_profile else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        with self._lock:
            return self.active_profile.get_embeddings(texts, mode=mode) if self.active_profile else []

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        with self._lock:
            if not self.active_profile:
                return ""
            return self.active_profile.generate_text(prompt, max_tokens, schema, system_prompt=system_prompt)

    def can_synthesize(self) -> bool:
        with self._lock:
            return self.active_profile.can_synthesize() if self.active_profile else False

    def can_feedback_check(self) -> bool:
        with self._lock:
            return self.active_profile.can_feedback_check() if self.active_profile else False

    def has_translator_pass(self) -> bool:
        with self._lock:
            return self.active_profile.has_translator_pass() if self.active_profile else False

    def has_semantic_compiler(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_semantic_compiler", lambda: False)() if self.active_profile else False

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "compile_semantic_intent"):
                return self.active_profile.compile_semantic_intent(prompt, schema=schema, system_prompt=system_prompt)
            return {"effective_prompt": prompt}

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        with self._lock:
            return self.active_profile.feedback_check(failing_code, traceback_error) if self.active_profile else failing_code

    @property
    def embedding_dimension(self) -> int:
        with self._lock:
            return self.active_profile.embedding_dimension if self.active_profile else 384
