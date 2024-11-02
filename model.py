from dataclasses import dataclass
import logging
from ollama import chat
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for Ollama model parameters."""

    model_name: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    max_retries: int = 3


class OllamaError(Exception):
    """Base exception class for Ollama-related errors."""

    pass


class MaxRetriesExceededError(OllamaError):
    """Raised when max retry attempts are exhausted."""

    pass


class EmptyResponseError(OllamaError):
    """Raised when Ollama returns an empty or invalid response."""

    pass


class OllamaModel:
    """Client for interacting with Ollama models."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama model client.

        Args:
            model_name: Name of the Ollama model to use
            temp: Temperature for generation (0.0-1.0)
            top_p: Top-p sampling parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum retry attempts on failure
        """
        self.config = ModelConfig(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def _create_chat_options(self) -> dict[str, float | int]:
        """Create options dictionary for chat API call."""

        return {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

    def _create_messages(self, prompt: str) -> list[dict[str, str]]:
        """Create messages list for chat API call."""

        return [{"role": "user", "content": prompt}]

    def _validate_response(self, response: dict[str, Any]) -> str:
        """
        Validate and extract content from response.

        Args:
            response: Raw response from Ollama API

        Returns:
            Extracted response content

        Raises:
            EmptyResponseError: If response content is empty or invalid
        """

        content = response.get("message", {}).get("content")

        if not content:
            logger.error(f"Empty or invalid response received: {response}")
            raise EmptyResponseError("No content in response")

        return content

    def _make_chat_request(self, prompt: str) -> str:
        """
        Make a single chat request to Ollama.

        Args:
            prompt: Input prompt for the model

        Returns:
            Model's response content

        Raises:
            EmptyResponseError: If response is empty or invalid
            Exception: For other API errors
        """

        response = chat(
            model=self.config.model_name,
            messages=self._create_messages(prompt),
            options=self._create_chat_options(),
        )

        return self._validate_response(response)

    def predict(self, prompt: str) -> str:
        """
        Generate a prediction for the given prompt with retry logic.

        Args:
            prompt: Input prompt for the model

        Returns:
            Model's response content

        Raises:
            MaxRetriesExceededError: If max retries are exceeded
        """

        for attempt in range(self.config.max_retries):
            try:
                return self._make_chat_request(prompt)
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed: {str(e)}"
                )
                if attempt == self.config.max_retries - 1:
                    logger.error("Max retries exceeded")
                    raise MaxRetriesExceededError(
                        f"Failed to get response after {self.config.max_retries} retries"
                    ) from e

        # This shouldn't be reached due to the raise in the loop
        raise MaxRetriesExceededError("Failed to get response after max retries")
