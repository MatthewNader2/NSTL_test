"""
NSTL Centralized Configuration
All environment-specific paths and settings should be defined here
with environment variable overrides for portability.
"""
import os
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
TREES_DIR = os.environ.get(
    "NSTL_TREES_DIR",
    str(PROJECT_ROOT / "trees")
)

HARVESTS_DIR = os.environ.get(
    "NSTL_HARVESTS_DIR",
    str(PROJECT_ROOT / "harvests")
)

LOGS_DIR = os.environ.get(
    "NSTL_LOGS_DIR",
    str(PROJECT_ROOT / "logs")
)

MODELS_DIR = os.environ.get(
    "NSTL_MODELS_DIR",
    str(PROJECT_ROOT / "models")
)

# LLM server paths
LLAMA_SERVER_EXE = os.environ.get(
    "NSTL_LLAMA_SERVER",
    ""  # Must be set per-machine or discovered at runtime
)

# Default model path (relative to MODELS_DIR)
DEFAULT_GGUF_MODEL = os.environ.get(
    "NSTL_DEFAULT_MODEL",
    os.path.join(MODELS_DIR, "llms", "Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
)

# Server config
API_HOST = os.environ.get("NSTL_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("NSTL_API_PORT", "58102"))

# CORS allowed origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    f"http://localhost:{API_PORT}",
    f"http://127.0.0.1:{API_PORT}",
]

# ---------------------------------------------------------------------------
# Scoring & Routing Thresholds (VIO-50/51 fix: centralized, configurable)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = float(os.environ.get("NSTL_SIMILARITY_THRESHOLD", "0.25"))
MIN_CONFIDENCE = float(os.environ.get("NSTL_MIN_CONFIDENCE", "0.30"))
TUNNELING_MARGIN = float(os.environ.get("NSTL_TUNNELING_MARGIN", "0.15"))
MACRO_THRESHOLD = float(os.environ.get("NSTL_MACRO_THRESHOLD", "0.40"))
TYPE_MISMATCH_DISCOUNT = float(os.environ.get("NSTL_TYPE_MISMATCH_DISCOUNT", "0.001"))

# Domain affinity factors for scoring
DOMAIN_MATCH_FACTOR = float(os.environ.get("NSTL_DOMAIN_MATCH", "1.0"))
DOMAIN_NEUTRAL_FACTOR = float(os.environ.get("NSTL_DOMAIN_NEUTRAL", "0.5"))
DOMAIN_CONFLICT_FACTOR = float(os.environ.get("NSTL_DOMAIN_CONFLICT", "0.01"))

import sys

def find_llama_server() -> str:
    """Attempts to find llama-server executable if not explicitly configured."""
    if LLAMA_SERVER_EXE and os.path.exists(LLAMA_SERVER_EXE):
        return LLAMA_SERVER_EXE
    
    bin_name = "llama-server.exe" if sys.platform.startswith("win") else "llama-server"
    
    # Check common locations
    tools_dir = PROJECT_ROOT / "tools" / "llama-cpp"
    if tools_dir.exists():
        for p in tools_dir.glob(f"**/{bin_name}"):
            if p.is_file():
                return str(p)

    search_paths = [
        tools_dir / bin_name,
        Path(MODELS_DIR) / bin_name,
        Path(MODELS_DIR) / "llama-server.exe",
        Path(MODELS_DIR) / "llama-server",
    ]
    
    for p in search_paths:
        if p.exists():
            return str(p)
    
    # Fall back to PATH
    import shutil
    found = shutil.which(bin_name) or shutil.which("llama-server")
    if found:
        return found
    
    return ""
