"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Synthetica-Core."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Synthetica-Core", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    exa_api_key: str = Field(default="", alias="EXA_API_KEY")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    together_api_key: str = Field(default="", alias="TOGETHER_API_KEY")

    # Module 2 — model cascade (fast 8B → dense 70B)
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="LLM_BASE_URL",
    )
    llm_model_fast: str = Field(
        default="llama-3.1-8b-instant",
        alias="LLM_MODEL_FAST",
    )
    llm_model_dense: str = Field(
        default="llama-3.3-70b-versatile",
        alias="LLM_MODEL_DENSE",
    )

    # Module 5 — deep arbitration models
    arbitration_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="ARBITRATION_BASE_URL",
    )
    arbitration_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="ARBITRATION_MODEL",
    )

    # Module 1 — Matryoshka projection dims (nested truncation)
    matryoshka_dims: str = Field(default="64,128,256,768", alias="MATRYOSHKA_DIMS")

    # Module 4 — entropy gate: H(R_q) ≤ threshold → fast path
    entropy_threshold: float = Field(default=0.55, alias="ENTROPY_THRESHOLD")

    # CAMMR blend
    cammr_lambda: float = Field(default=0.7, alias="CAMMR_LAMBDA")

    # Research path — latency / quality knobs (cooked path)
    research_top_n: int = Field(default=4, ge=1, le=20, alias="RESEARCH_TOP_N")
    research_extract_docs: int = Field(
        default=3,
        ge=1,
        le=20,
        alias="RESEARCH_EXTRACT_DOCS",
    )
    research_max_chars: int = Field(
        default=4000,
        ge=500,
        le=20000,
        alias="RESEARCH_MAX_CHARS",
    )
    research_min_relevance: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        alias="RESEARCH_MIN_RELEVANCE",
    )
    claim_cache_ttl_seconds: float = Field(
        default=3600.0,
        ge=0.0,
        alias="CLAIM_CACHE_TTL_SECONDS",
    )

    @property
    def llm_api_key(self) -> str:
        """Prefer Groq, fall back to Together."""
        return self.groq_api_key or self.together_api_key

    @property
    def matryoshka_dim_list(self) -> list[int]:
        return [int(x.strip()) for x in self.matryoshka_dims.split(",") if x.strip()]

    # Backward-compatible alias used by older extractor code paths
    @property
    def llm_model(self) -> str:
        return self.llm_model_fast


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
