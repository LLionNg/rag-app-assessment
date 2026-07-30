from __future__ import annotations

from src.core.config import LLMConfig
from src.core.exceptions import ConfigError
from src.llm.azure_openai import AzureOpenAIProvider
from src.llm.base import LLMProvider
from src.llm.mock import MockLLMProvider
from src.llm.openai_compatible import OpenAICompatibleProvider
from src.llm.responses_api import ResponsesAPIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "bbl_gateway": ResponsesAPIProvider,
    "azure_openai": AzureOpenAIProvider,
    "openai": OpenAICompatibleProvider,
    "mock": MockLLMProvider,
}


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    try:
        provider_cls = _PROVIDERS[config.provider]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ConfigError(
            f"Unknown llm.provider '{config.provider}' (available: {known})"
        ) from None
    return provider_cls(config)
