"""
Unit tests for custom exception classes.
Tests exception hierarchy, user messages, and error propagation.
"""

import pytest

from app.core.exceptions import (
    RAGError,
    RAGQueryError,
    RAGIndexError,
    EmbeddingError,
    LLMError,
)


class TestRAGError:
    """Tests for base RAGError class."""

    def test_rag_error_with_default_user_message(self):
        """Test RAGError uses default user message when not provided."""
        error = RAGError("Internal error details")

        assert str(error) == "Internal error details"
        assert error.user_message == "An error occurred while processing your request."

    def test_rag_error_with_custom_user_message(self):
        """Test RAGError accepts custom user message."""
        error = RAGError("Internal details", user_message="Custom user message")

        assert str(error) == "Internal details"
        assert error.user_message == "Custom user message"

    def test_rag_error_is_exception(self):
        """Test RAGError inherits from Exception."""
        error = RAGError("test")
        assert isinstance(error, Exception)

    def test_rag_error_can_be_raised_and_caught(self):
        """Test RAGError can be raised and caught properly."""
        with pytest.raises(RAGError) as exc_info:
            raise RAGError("internal", user_message="user facing")

        assert exc_info.value.user_message == "user facing"


class TestRAGQueryError:
    """Tests for RAGQueryError class."""

    def test_query_error_default_user_message(self):
        """Test RAGQueryError has appropriate default message."""
        error = RAGQueryError("Query failed: timeout")

        assert str(error) == "Query failed: timeout"
        assert error.user_message == "Unable to process your query. Please try again."

    def test_query_error_custom_user_message(self):
        """Test RAGQueryError accepts custom user message."""
        error = RAGQueryError(
            "Internal: vector store unreachable",
            user_message="Search is temporarily unavailable.",
        )

        assert "vector store" in str(error)
        assert error.user_message == "Search is temporarily unavailable."

    def test_query_error_inherits_from_rag_error(self):
        """Test RAGQueryError inherits from RAGError."""
        error = RAGQueryError("test")
        assert isinstance(error, RAGError)
        assert isinstance(error, Exception)

    def test_query_error_caught_as_rag_error(self):
        """Test RAGQueryError can be caught as RAGError."""
        with pytest.raises(RAGError):
            raise RAGQueryError("query failed")


class TestRAGIndexError:
    """Tests for RAGIndexError class."""

    def test_index_error_default_user_message(self):
        """Test RAGIndexError has appropriate default message."""
        error = RAGIndexError("Embedding API rate limited")

        assert str(error) == "Embedding API rate limited"
        assert error.user_message == "Unable to index document. Please try again."

    def test_index_error_custom_user_message(self):
        """Test RAGIndexError accepts custom user message."""
        error = RAGIndexError(
            "Document too large: 500MB",
            user_message="Document exceeds size limit.",
        )

        assert "500MB" in str(error)
        assert error.user_message == "Document exceeds size limit."

    def test_index_error_inherits_from_rag_error(self):
        """Test RAGIndexError inherits from RAGError."""
        error = RAGIndexError("test")
        assert isinstance(error, RAGError)


class TestEmbeddingError:
    """Tests for EmbeddingError class."""

    def test_embedding_error_default_user_message(self):
        """Test EmbeddingError has appropriate default message."""
        error = EmbeddingError("OpenAI API key invalid")

        assert str(error) == "OpenAI API key invalid"
        assert error.user_message == "Unable to process text embeddings."

    def test_embedding_error_custom_user_message(self):
        """Test EmbeddingError accepts custom user message."""
        error = EmbeddingError(
            "Token limit exceeded",
            user_message="Text is too long to process.",
        )

        assert error.user_message == "Text is too long to process."

    def test_embedding_error_inherits_from_rag_error(self):
        """Test EmbeddingError inherits from RAGError."""
        error = EmbeddingError("test")
        assert isinstance(error, RAGError)


class TestLLMError:
    """Tests for LLMError class."""

    def test_llm_error_default_user_message(self):
        """Test LLMError has appropriate default message."""
        error = LLMError("GPT-4 context window exceeded")

        assert str(error) == "GPT-4 context window exceeded"
        assert error.user_message == "Unable to generate response. Please try again."

    def test_llm_error_custom_user_message(self):
        """Test LLMError accepts custom user message."""
        error = LLMError(
            "Model not available",
            user_message="AI service is temporarily unavailable.",
        )

        assert error.user_message == "AI service is temporarily unavailable."

    def test_llm_error_inherits_from_rag_error(self):
        """Test LLMError inherits from RAGError."""
        error = LLMError("test")
        assert isinstance(error, RAGError)


class TestExceptionChaining:
    """Tests for exception chaining behavior."""

    def test_exception_chaining_preserves_cause(self):
        """Test that exception chaining preserves original cause."""
        original = ValueError("Original cause")

        try:
            try:
                raise original
            except ValueError as e:
                raise RAGQueryError("Wrapped error") from e
        except RAGQueryError as e:
            assert e.__cause__ is original
            assert isinstance(e.__cause__, ValueError)

    def test_user_message_does_not_leak_internal_details(self):
        """Test that user_message doesn't contain internal details."""
        sensitive_details = "API key: sk-abc123, path: /internal/data"
        error = RAGQueryError(sensitive_details)

        # Internal message has details
        assert "sk-abc123" in str(error)

        # User message is safe
        assert "sk-abc123" not in error.user_message
        assert "/internal" not in error.user_message


class TestExceptionHierarchy:
    """Tests for exception hierarchy catching behavior."""

    def test_catch_all_rag_errors(self):
        """Test that all specific errors can be caught as RAGError."""
        errors = [
            RAGQueryError("query"),
            RAGIndexError("index"),
            EmbeddingError("embedding"),
            LLMError("llm"),
        ]

        for error in errors:
            with pytest.raises(RAGError):
                raise error

    def test_specific_catch_does_not_catch_siblings(self):
        """Test that catching specific error doesn't catch siblings."""
        with pytest.raises(RAGIndexError):
            try:
                raise RAGIndexError("index error")
            except RAGQueryError:
                pytest.fail("RAGQueryError should not catch RAGIndexError")
