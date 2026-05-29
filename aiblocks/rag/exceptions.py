"""Custom exception hierarchy for the aiblocks RAG module."""


class AIBlocksError(Exception):
    """Base exception for all aiblocks errors."""


class ConfigurationError(AIBlocksError, ValueError):
    """Raised when module configuration is invalid."""


class IngestionError(AIBlocksError, RuntimeError):
    """Raised when document ingestion fails."""


class RetrievalError(AIBlocksError, RuntimeError):
    """Raised when retrieval fails."""


class GenerationError(AIBlocksError, RuntimeError):
    """Raised when generation fails."""


class ModelNotAvailableError(AIBlocksError, ConnectionError):
    """Raised when a model or external service is unreachable."""
