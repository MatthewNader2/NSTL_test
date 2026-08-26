"""
src/config.py - Centralized Configuration for NSTL
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TREES_DIR = os.environ.get("NSTL_TREES_DIR", str(PROJECT_ROOT / "trees"))
HARVESTS_DIR = os.environ.get("NSTL_HARVESTS_DIR", str(PROJECT_ROOT / "harvests"))
LOGS_DIR = os.environ.get("NSTL_LOGS_DIR", str(PROJECT_ROOT / "logs"))
MODELS_DIR = os.environ.get("NSTL_MODELS_DIR", str(PROJECT_ROOT / "models"))

API_HOST = os.environ.get("NSTL_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("NSTL_API_PORT", "58102"))

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    f"http://localhost:{API_PORT}",
    f"http://127.0.0.1:{API_PORT}",
]
