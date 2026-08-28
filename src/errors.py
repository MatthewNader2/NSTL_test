"""
src/errors.py - NSTL Structured Error Hierarchy
Provides typed exceptions for clear error propagation instead of silent `except: pass`.
"""


class NSTLError(Exception):
    """Base exception for all NSTL errors."""
    pass


# === Routing Errors ===
class RoutingError(NSTLError):
    """Failed to find a valid path through the lattice."""
    pass


class NoPathFoundError(RoutingError):
    """A* search exhausted all candidates without reaching the goal."""
    pass


class InsufficientCandidatesError(RoutingError):
    """Too few candidates to form a viable path."""
    pass


# === Synthesis Errors ===
class SynthesisError(NSTLError):
    """Failed to synthesize code from lattice cells."""
    pass


class UnificationError(SynthesisError):
    """Typestate unification failed during code emission."""
    pass


class PlaceholderResolutionError(SynthesisError):
    """A code template placeholder could not be resolved."""
    pass


class TemplateValidationError(SynthesisError):
    """A synthesized code template failed AST validation."""
    pass


# === Execution Errors ===
class ExecutionError(NSTLError):
    """Sandbox execution failed."""
    pass


class SandboxTimeoutError(ExecutionError):
    """Execution exceeded the configured time limit."""
    pass


class SandboxSecurityError(ExecutionError):
    """Execution attempted to use a blocked resource."""
    pass


# === Model Errors ===
class ModelError(NSTLError):
    """Model loading or inference failed."""
    pass


class ModelNotLoadedError(ModelError):
    """No model profile is currently active."""
    pass


class EmbeddingError(ModelError):
    """Embedding generation failed."""
    pass


class LLMInferenceError(ModelError):
    """LLM text generation failed."""
    pass


# === RAG Errors ===
class RAGError(NSTLError):
    """Retrieval-Augmented Generation failed."""
    pass


class IndexError(RAGError):
    """FAISS index operation failed."""
    pass


class FetchError(RAGError):
    """External documentation fetch failed."""
    pass


# === Configuration Errors ===
class ConfigurationError(NSTLError):
    """Configuration validation failed."""
    pass


class LatticeError(NSTLError):
    """Lattice database or topology operation failed."""
    pass
