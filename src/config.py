"""
src/config.py - Centralized Configuration for NSTL
Uses Pydantic BaseSettings for validated, type-safe configuration with .env file support.
"""
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import Field, field_validator
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

    # Server & Networking
    api_host: str = Field(default="127.0.0.1", description="Host interface to bind the API server.")
    api_port: int = Field(default=58102, ge=1, le=65535, description="Port to bind the API server.")
    cors_origins: List[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Configurable via NSTL_CORS_ORIGINS (comma-separated or JSON list).",
    )

    # Sandbox
    sandbox_enabled: bool = Field(default=False, description="Whether sandbox isolation is enabled.")
    sandbox_timeout: float = Field(default=5.0, gt=0, description="Sandbox timeout in seconds.")
    sandbox_workers: int = Field(default=2, ge=1, description="Number of sandbox worker processes.")
    sandbox_max_memory_mb: int = Field(default=1024, ge=64, description="Sandbox memory limit in MB.")
    sandbox_max_cpu_seconds: int = Field(default=5, ge=1, description="Sandbox CPU runtime limit in seconds.")

    # Inference & Similarity
    llm_context_length: int = Field(default=4096, ge=512)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    similarity_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic candidate similarity matching.",
    )

    # Macro-goal routing & Topology Planning
    macros_enabled: bool = True
    topology_mode: str = "frontier"
    use_ir_compiler: bool = False
    require_coverage_floor: bool = False
    dev_mode: bool = False
    planner_config: Optional[Path] = None
    planner_time_budget_ms: float = Field(
        default=10000.0,
        ge=250.0,
        description="Wall-clock budget for a single planning search. When expired, "
        "the beam returns the best-so-far valid paths instead of running unbounded.",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                import json
                try:
                    return json.loads(stripped)
                except Exception:
                    pass
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value or []

    def model_post_init(self, __context: Any) -> None:
        """Resolve default paths and dynamic defaults after construction."""
        if self.trees_dir is None:
            self.trees_dir = self.project_root / "trees"
        if self.harvests_dir is None:
            self.harvests_dir = self.project_root / "harvests"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"

        if getattr(self, "planner_config", None) is None:
            for cand in (
                self.project_root / "config" / "planner.yaml",
                self.project_root / "config" / "planner.json",
            ):
                if cand.exists():
                    self.planner_config = cand
                    break

        if not self.cors_origins:
            self.cors_origins = [
                f"http://{self.api_host}:{self.api_port}",
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
SIMILARITY_THRESHOLD = settings.similarity_threshold
