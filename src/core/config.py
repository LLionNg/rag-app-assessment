from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.core.exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path("config.yml")
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class AppConfig(BaseModel):
    name: str = "rag-app-assessment"
    environment: str = "local"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Path | None = None
    rotation: str = "10 MB"
    retention: str = "7 days"
    serialize: bool = False


class LocalModelOptions(BaseModel):
    """Knobs for models that run in-process rather than behind an API."""

    backend: Literal["sentence_transformers", "flag_embedding"] = (
        "sentence_transformers"
    )
    use_fp16: bool = False
    max_length: int = Field(default=8192, gt=0)
    batch_size: int = Field(default=12, gt=0)
    device: str | None = None
    cache_dir: Path | None = None


class ProviderConfig(BaseModel):
    """Connection settings for one LLM or embedding provider.

    API keys are never stored here: `api_key_env` names the environment
    variable the provider reads at construction time.
    """

    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    token_param: Literal["max_tokens", "max_completion_tokens", "max_output_tokens"] = (
        "max_tokens"
    )
    supports_temperature: bool = True
    # Gateways in front of Azure OpenAI authenticate with `api-key` rather than
    # the SDK default of `Authorization: Bearer`.
    auth_header: str = "api-key"
    # Reasoning models bill hidden reasoning tokens; "low" cuts them sharply.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # Each parallel tool call replays its whole result into the next request, so
    # under a tight token budget these two caps matter more than they look.
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Expected embedding width, enforced on every batch. BGE-M3 dense is 1024.
    dimensions: int | None = Field(default=None, gt=0)
    options: LocalModelOptions = Field(default_factory=LocalModelOptions)

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


class _ProviderSelection(BaseModel):
    provider: str | None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    def active(self) -> ProviderConfig:
        if not self.provider:
            raise ConfigError("No provider selected for this component")
        try:
            return self.providers[self.provider]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "none"
            raise ConfigError(
                f"Provider '{self.provider}' has no settings block (known: {known})"
            ) from None


class LLMConfig(_ProviderSelection):
    provider: str = "mock"
    temperature: float | None = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


class VectorStoreConfig(BaseModel):
    """Where the persisted embedding index lives. No database required."""

    enabled: bool = True
    path: Path = Path("data/index")
    name: str = "knowledge_base"
    refresh: bool = False


class EmbeddingsConfig(_ProviderSelection):
    provider: str | None = None
    batch_size: int = Field(default=32, gt=0)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    @property
    def enabled(self) -> bool:
        return bool(self.provider)


class ChunkingConfig(BaseModel):
    strategy: Literal["paragraph", "fixed"] = "paragraph"
    max_chars: int = 900
    min_chars: int = 80
    overlap_chars: int = 120


class KnowledgeBaseConfig(BaseModel):
    path: Path = Path("data/knowledge_base.txt")
    encoding: str = "utf-8"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class KeywordRetrievalConfig(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class SemanticRetrievalConfig(BaseModel):
    min_similarity: float = 0.2


class HybridRetrievalConfig(BaseModel):
    rrf_k: int = 60
    weights: dict[str, float] = Field(
        default_factory=lambda: {"keyword": 0.5, "semantic": 0.5}
    )


class RetrievalConfig(BaseModel):
    strategy: Literal["keyword", "semantic", "hybrid"] = "keyword"
    top_k: int = 4
    min_score: float = 0.0
    keyword: KeywordRetrievalConfig = Field(default_factory=KeywordRetrievalConfig)
    semantic: SemanticRetrievalConfig = Field(default_factory=SemanticRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)


class DataRetrieverAgentConfig(BaseModel):
    name: str = "Data Retriever"
    max_tool_iterations: int = 3
    max_snippets: int = 6


class ReportGeneratorAgentConfig(BaseModel):
    name: str = "Report Generator"
    max_snippet_chars: int = 1200


class AgentsConfig(BaseModel):
    data_retriever: DataRetrieverAgentConfig = Field(
        default_factory=DataRetrieverAgentConfig
    )
    report_generator: ReportGeneratorAgentConfig = Field(
        default_factory=ReportGeneratorAgentConfig
    )


class OrchestrationConfig(BaseModel):
    engine: Literal["sequential"] = "sequential"


class DemoConfig(BaseModel):
    queries: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load `config.yml`, expanding `${VAR}` / `${VAR:-default}` placeholders."""
    load_dotenv(override=False)

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Expected a mapping at the root of {config_path}")

    try:
        return Settings.model_validate(_expand_env(raw))
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{exc}") from exc


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return _ENV_PLACEHOLDER.sub(
            lambda m: os.getenv(m.group(1), m.group(2) or ""), value
        )
    return value
