"""
Abstract base class for LLM clients.

Defines the common interface and shared utilities for all LLM provider
implementations (Anthropic, OpenAI, xAI, etc.).

Subclasses must implement:
    - prompt_llm(): Send a prompt and return the response with metadata.

See AnthropicClient for a reference implementation.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM API clients.

    Provides shared utilities for JSON extraction, response validation,
    and token counting. Subclasses implement provider-specific API calls.
    """

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract JSON content from response, handling markdown code blocks.

        Args:
            response_content: Raw response content from LLM.

        Returns:
            JSON string extracted from code blocks, or the original content.
        """
        if not response_content:
            return response_content

        response_content = response_content.strip()

        # Handle ```json ... ``` blocks
        if response_content.startswith("```json") and response_content.endswith("```"):
            lines = response_content.split("\n")
            if len(lines) > 2:
                return "\n".join(lines[1:-1]).strip()

        # Handle ``` ... ``` blocks (without json specifier)
        elif response_content.startswith("```") and response_content.endswith("```"):
            lines = response_content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1]).strip()
                if content.startswith("{") and content.endswith("}"):
                    return content

        return response_content

    def _validate_response(
        self,
        response_content: str,
        response_model: Type[BaseModel],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Parse JSON and validate against a Pydantic model.

        Args:
            response_content: JSON string from the LLM.
            response_model: Pydantic model class for validation.

        Returns:
            Tuple of (validated model instance, raw dict).

        Raises:
            ValueError: If JSON parsing or validation fails.
        """
        try:
            response_json = json.loads(response_content)
            validated_response = response_model.model_validate(response_json)
            return validated_response, response_json
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse response as JSON: {e}\nResponse: {response_content}"
            )
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Override in subclasses for provider-specific tokenizers.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return int(len(text.split()) * 1.3)

    @abstractmethod
    def prompt_llm(
        self,
        model: str,
        message_history: List[Dict[str, Any]] = None,
        question: str = None,
        base64_images: List[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        retries: int = 20,
        retry_delay: int = 5,
    ) -> Any:
        """
        Send a prompt to the LLM and return the response.

        Implementations must return a dict with the following structure:

        Without response_model::

            {
                "response": str,          # extracted answer text
                "metadata": {
                    "client_type": str,    # e.g. "anthropic", "azure_openai"
                    "tokens": {
                        "prompt_tokens": int | None,
                        "completion_tokens": int | None,
                        "total_tokens": int | None,
                        "available": bool,
                    },
                    "model_info": {
                        "model_used": str,
                        "response_id": str | None,
                        "finish_reason": str | None,
                        "created": int | None,
                    },
                    "timing": {
                        "request_start": str,   # ISO 8601
                        "response_end": str,
                        "duration_ms": int,
                    },
                    "request_config": { ... },
                    "processing": { ... },
                }
            }

        With response_model::

            {
                "validated_response": BaseModel,  # validated Pydantic instance
                "raw_json": dict,                 # raw parsed JSON
                "metadata": { ... },              # same structure as above
            }

        Args:
            model: Model name or deployment ID.
            message_history: List of message dicts with 'role' and 'content'.
            question: Single question string (alternative to message_history).
            base64_images: Optional list of base64-encoded images.
            response_model: Optional Pydantic model for structured output.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            retries: Number of retry attempts on failure.
            retry_delay: Base delay in seconds between retries.

        Returns:
            Dict with response content and metadata.

        Raises:
            ValueError: If neither message_history nor question is provided.
            Exception: If all retries are exhausted.
        """
        ...
