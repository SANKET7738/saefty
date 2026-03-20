"""
Anthropic API Client using the official Anthropic SDK.

Reference implementation of BaseLLMClient for the Anthropic (Claude) API.
Use this as a template for implementing other LLM provider clients.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from .base_client import BaseLLMClient

try:
    import anthropic
except ImportError:
    raise ImportError(
        "anthropic is required for Anthropic API. Install with: pip install anthropic"
    )


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client using the official SDK.

    Inherits shared utilities from BaseLLMClient and implements
    the Anthropic-specific API call logic.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 600):
        """
        Initialize Anthropic client with authentication.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                     ANTHROPIC_API_KEY env var.
            timeout: Request timeout in seconds (default: 600).
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set or api_key must be provided"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)

    def _convert_message_history_to_anthropic_format(
        self, message_history: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert standard message history format to Anthropic API format.

        Anthropic handles system messages separately from user/assistant messages,
        and has its own image format.

        Args:
            message_history: Standard message history with role/content dicts.

        Returns:
            Tuple of (anthropic messages, system prompt string or None).
        """
        anthropic_messages = []
        system_prompt = None

        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages separately
            if role == "system":
                if system_prompt is None:
                    system_prompt = ""
                else:
                    system_prompt += "\n\n"

                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        else:
                            text_parts.append(str(item))
                    system_prompt += " ".join(text_parts)
                else:
                    system_prompt += str(content)
                continue

            # Handle content that might be a list (for multimodal)
            if isinstance(content, list):
                anthropic_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            anthropic_content.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                try:
                                    header, data = image_url.split(",", 1)
                                    media_type = header.split(";")[0].split(":")[1]
                                    anthropic_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data,
                                            },
                                        }
                                    )
                                except Exception as e:
                                    print(f"Warning: Failed to process image: {e}")
                    else:
                        anthropic_content.append({"type": "text", "text": str(item)})

                if (
                    len(anthropic_content) == 1
                    and anthropic_content[0].get("type") == "text"
                ):
                    content = anthropic_content[0]["text"]
                else:
                    content = anthropic_content

            anthropic_role = "assistant" if role == "assistant" else "user"
            anthropic_messages.append({"role": anthropic_role, "content": content})

        return anthropic_messages, system_prompt

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
        Send a prompt to Anthropic Claude and get the response.

        Args:
            model: Model name (e.g. "claude-sonnet-4-20250514").
            message_history: List of message dicts (alternative to question).
            question: Single question string (alternative to message_history).
            base64_images: Optional list of base64-encoded images.
            response_model: Optional Pydantic model for structured output.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            retries: Number of retries on failure.
            retry_delay: Delay between retries in seconds.

        Returns:
            Dict with response content and metadata.
        """
        attempts = 0
        last_error = None

        while attempts <= retries:
            try:
                request_start_time = datetime.now()

                # Prepare messages and system prompt
                if message_history:
                    messages, system_prompt = (
                        self._convert_message_history_to_anthropic_format(
                            message_history
                        )
                    )
                elif question:
                    content = question
                    if base64_images:
                        content_list = [{"type": "text", "text": question}]
                        for base64_image in base64_images:
                            content_list.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image,
                                    },
                                }
                            )
                        content = content_list
                    messages = [{"role": "user", "content": content}]
                    system_prompt = None
                else:
                    raise ValueError(
                        "Either message_history or question must be provided"
                    )

                api_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system_prompt:
                    api_params["system"] = system_prompt

                response = self.client.messages.create(**api_params)
                request_end_time = datetime.now()

                # Extract response text
                response_content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        response_content += content_block.text

                # Build metadata
                system_prompt_tokens = 0
                if system_prompt:
                    system_prompt_tokens = self.count_tokens(system_prompt)
                elif message_history:
                    for msg in message_history:
                        if msg.get("role") == "system":
                            system_prompt_tokens += self.count_tokens(
                                str(msg.get("content", ""))
                            )

                metadata = {
                    "client_type": "anthropic",
                    "tokens": {
                        "prompt_tokens": response.usage.input_tokens
                        if hasattr(response, "usage")
                        else None,
                        "completion_tokens": response.usage.output_tokens
                        if hasattr(response, "usage")
                        else None,
                        "total_tokens": (
                            response.usage.input_tokens
                            + response.usage.output_tokens
                        )
                        if hasattr(response, "usage")
                        else None,
                        "available": hasattr(response, "usage"),
                    },
                    "model_info": {
                        "model_used": response.model,
                        "response_id": response.id,
                        "finish_reason": response.stop_reason,
                        "created": None,
                    },
                    "timing": {
                        "request_start": request_start_time.isoformat(),
                        "response_end": request_end_time.isoformat(),
                        "duration_ms": int(
                            (
                                request_end_time - request_start_time
                            ).total_seconds()
                            * 1000
                        ),
                    },
                    "request_config": {
                        "temperature": temperature,
                        "top_p": None,
                        "max_tokens": max_tokens,
                        "retries_attempted": attempts,
                    },
                    "processing": {
                        "system_prompt_tokens": system_prompt_tokens,
                        "raw_response_length": len(response_content)
                        if response_content
                        else 0,
                    },
                }

                # Validate if response_model provided
                if response_model:
                    clean_json = self._extract_json_from_response(response_content)
                    validated_response, response_json = self._validate_response(
                        clean_json, response_model
                    )
                    metadata["processing"].update(
                        {
                            "json_parse_success": True,
                            "extracted_answer_length": len(str(validated_response)),
                        }
                    )
                    return {
                        "validated_response": validated_response,
                        "raw_json": response_json,
                        "metadata": metadata,
                    }

                # Try to extract JSON answer
                extracted_answer = response_content
                json_parse_success = True
                try:
                    clean_json = self._extract_json_from_response(response_content)
                    response_json = json.loads(clean_json)
                    extracted_answer = response_json.get("answer", response_content)
                except (json.JSONDecodeError, TypeError):
                    json_parse_success = False

                metadata["processing"].update(
                    {
                        "json_parse_success": json_parse_success,
                        "extracted_answer_length": len(extracted_answer)
                        if extracted_answer
                        else 0,
                    }
                )

                return {"response": extracted_answer, "metadata": metadata}

            except Exception as e:
                print(f"Anthropic API attempt {attempts + 1} failed: {e}")
                last_error = e
                attempts += 1

                if attempts <= retries:
                    backoff = min(retry_delay * (2 ** (attempts - 1)), 120)
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    print(
                        f"All {retries} retries exhausted. Last error: {last_error}"
                    )
                    raise last_error
