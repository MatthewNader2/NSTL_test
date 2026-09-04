"""
src/config.py - Centralized Configuration for NSTL
Uses Pydantic BaseSettings for validated, type-safe configuration with .env file support.
"""
from pathlib import Path
from typing import Any, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class NSTLSettings(BaseSettings):
    """All NSTL configuration values, resolved from environment variables with NSTL_ prefix."""
    model_config = SettingsConfigDict(
        env_prefix="NSTL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    trees_dir: Optional[Path] = None
    harvests_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    models_dir: Optional[Path] = None

    # Server
    api_host: str = "127.0.0.1"
    api_port: int = 58102

    # Sandbox
    sandbox_timeout: float = 5.0
    sandbox_workers: int = 2
    sandbox_max_memory_mb: int = 1024
    sandbox_max_cpu_seconds: int = 5

    # Inference
    llm_context_length: int = 2048
    llm_temperature: float = 0.1
    llm_top_p: float = 0.95

    def model_post_init(self, __context) -> None:
        """Resolve default paths relative to project_root after construction."""
        if self.trees_dir is None:
            self.trees_dir = self.project_root / "trees"
        if self.harvests_dir is None:
            self.harvests_dir = self.project_root / "harvests"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"

    @property
    def cors_origins(self) -> List[str]:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            f"http://localhost:{self.api_port}",
            f"http://127.0.0.1:{self.api_port}",
        ]


# Singleton instance — importable as `from config import settings`
settings = NSTLSettings()

# Backward-compatible aliases (preserves existing import contracts)
PROJECT_ROOT = settings.project_root
TREES_DIR = str(settings.trees_dir)
HARVESTS_DIR = str(settings.harvests_dir)
LOGS_DIR = str(settings.logs_dir)
MODELS_DIR = str(settings.models_dir)
API_HOST = settings.api_host
API_PORT = settings.api_port
CORS_ORIGINS = settings.cors_origins
SIMILARITY_THRESHOLD = 0.25

def find_llama_server():
    return None
