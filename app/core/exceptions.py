"""Custom exceptions for the RAG application.

These exceptions provide structured error handling that:
- Separates internal details from user-facing messages
- Enables proper error propagation for monitoring
- Maintains security by not leaking implementation details
"""


class RAGError(Exception):
    """Base exception for RAG-related errors."""

    def __init__(self, message: str, user_message: str | None = None):
        """
        Initialize RAG error.

        Args:
            message: Internal error message for logging/debugging
            user_message: Safe message to show to users (defaults to generic message)
        """
        super().__init__(message)
        self.user_message = user_message or "An error occurred while processing your request."


class RAGQueryError(RAGError):
    """Error during RAG query processing."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(
            message,
            user_message or "Unable to process your query. Please try again.",
        )


class RAGIndexError(RAGError):
    """Error during document indexing."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(
            message,
            user_message or "Unable to index document. Please try again.",
        )


class EmbeddingError(RAGError):
    """Error generating embeddings."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(
            message,
            user_message or "Unable to process text embeddings.",
        )


class LLMError(RAGError):
    """Error calling the LLM."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(
            message,
            user_message or "Unable to generate response. Please try again.",
        )
