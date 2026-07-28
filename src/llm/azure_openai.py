from __future__ import annotations

from openai import AsyncAzureOpenAI

from src.core.exceptions import ConfigError
from src.llm.openai_compatible import OpenAICompatibleProvider


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI deployment (e.g. the gpt-5-mini deployment used for this test)."""

    def _build_client(self) -> AsyncAzureOpenAI:
        if not self.provider_config.endpoint:
            raise ConfigError("azure_openai provider requires `endpoint`")
        if not self.provider_config.api_version:
            raise ConfigError("azure_openai provider requires `api_version`")

        return AsyncAzureOpenAI(
            api_key=self._require_api_key(),
            azure_endpoint=self.provider_config.endpoint,
            api_version=self.provider_config.api_version,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )
