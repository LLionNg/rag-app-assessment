class RagAppError(Exception):
    """Base error for every failure raised by this application."""


class ConfigError(RagAppError):
    """Configuration is missing, malformed, or references an unknown component."""


class ProviderError(RagAppError):
    """An LLM or embedding provider could not fulfil a request."""


class KnowledgeBaseError(RagAppError):
    """The knowledge base could not be loaded or produced no usable chunks."""


class RetrievalError(RagAppError):
    """A retriever failed to index or to search."""
