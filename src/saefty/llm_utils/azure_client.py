"""
Azure OpenAI client — local-only, gitignored.

Uses AzureCliCredential for authentication with token caching.
Requires env vars: ENDPOINT_URL, DEPLOYMENT_NAME, API_VERSION
Optional: MANAGED_CLIENT_ID, REASONING_MODELS_LIST
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import AzureCliCredential
from pydantic import BaseModel, ValidationError

from .base_client import BaseLLMClient

load_dotenv()

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MANAGED_CLIENT_ID = os.getenv("MANAGED_CLIENT_ID")
API_VERSION = os.getenv("API_VERSION")
REASONING_MODELS_LIST = (
    os.getenv("REASONING_MODELS_LIST").split(",")
    if os.getenv("REASONING_MODELS_LIST")
    else []
)


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI client with AzureCliCredential and token caching."""

    def __init__(
        self,
        endpoint_url: str = ENDPOINT_URL,
        managed_client_id: str = MANAGED_CLIENT_ID,
        api_version: str = API_VERSION,
    ):
        self.endpoint = endpoint_url
        self.managed_client_id = managed_client_id
        self.api_version = api_version
        self.client = self._initialize_client()

    def _get_bearer_token_provider(self):
        try:
            credential = AzureCliCredential()
            credential.get_token("https://cognitiveservices.azure.com/.default")
        except Exception:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential(
                managed_identity_client_id=self.managed_client_id
            )

        _token_lock = threading.Lock()
        _cached = {"token": None, "expires_on": 0}

        def get_token():
            now = time.time()
            if _cached["token"] and now < _cached["expires_on"] - 300:
                return _cached["token"]
            with _token_lock:
                now = time.time()
                if _cached["token"] and now < _cached["expires_on"] - 300:
                    return _cached["token"]
                result = credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                _cached["token"] = result.token
                _cached["expires_on"] = result.expires_on
                return result.token

        return get_token

    def _initialize_client(self) -> AzureOpenAI:
        token_provider = self._get_bearer_token_provider()
        return AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

    def _serialize_logprobs(self, logprobs_content):
        if not logprobs_content:
            return None
        serialized = []
        try:
            for token_logprob in logprobs_content:
                token_data = {
                    "token": str(getattr(token_logprob, "token", "")),
                    "logprob": float(getattr(token_logprob, "logprob", 0.0)),
                    "bytes": getattr(token_logprob, "bytes", None),
                    "top_logprobs": [],
                }
                top = getattr(token_logprob, "top_logprobs", None)
                if top:
                    for tl in top:
                        token_data["top_logprobs"].append(
                            {
                                "token": str(getattr(tl, "token", "")),
                                "logprob": float(getattr(tl, "logprob", 0.0)),
                                "bytes": getattr(tl, "bytes", None),
                            }
                        )
                serialized.append(token_data)
        except Exception as e:
            print(f"Warning: Failed to serialize logprobs: {e}")
            return None
        return serialized

    def _extract_metadata(
        self,
        completion,
        request_start_time,
        request_end_time,
        messages,
        temperature,
        top_p,
        max_tokens,
        retries_attempted,
        model,
    ):
        try:
            system_prompt_tokens = 0
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt_tokens += self.count_tokens(
                        str(msg.get("content", ""))
                    )

            choice = completion.choices[0]
            is_reasoning_model = model in REASONING_MODELS_LIST
            logprobs_requested = not is_reasoning_model

            if hasattr(choice, "logprobs") and choice.logprobs is not None:
                content_logprobs = None
                if (
                    hasattr(choice.logprobs, "content")
                    and choice.logprobs.content
                ):
                    content_logprobs = self._serialize_logprobs(
                        choice.logprobs.content
                    )
                logprobs_data = {
                    "available": True,
                    "requested": logprobs_requested,
                    "content": content_logprobs,
                    "reasoning_model": is_reasoning_model,
                }
            else:
                logprobs_data = {
                    "available": False,
                    "requested": logprobs_requested,
                    "content": None,
                    "reasoning_model": is_reasoning_model,
                }

            return {
                "client_type": "azure_openai",
                "tokens": {
                    "prompt_tokens": completion.usage.prompt_tokens
                    if completion.usage
                    else None,
                    "completion_tokens": completion.usage.completion_tokens
                    if completion.usage
                    else None,
                    "total_tokens": completion.usage.total_tokens
                    if completion.usage
                    else None,
                    "available": completion.usage is not None,
                },
                "model_info": {
                    "model_used": completion.model,
                    "response_id": completion.id,
                    "finish_reason": choice.finish_reason,
                    "created": completion.created,
                },
                "timing": {
                    "request_start": request_start_time.isoformat(),
                    "response_end": request_end_time.isoformat(),
                    "duration_ms": int(
                        (request_end_time - request_start_time).total_seconds()
                        * 1000
                    ),
                },
                "request_config": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "retries_attempted": retries_attempted,
                    "logprobs_enabled": logprobs_requested,
                    "top_logprobs": 5 if logprobs_requested else None,
                    "is_reasoning_model": is_reasoning_model,
                },
                "processing": {
                    "system_prompt_tokens": system_prompt_tokens,
                },
                "logprobs": logprobs_data,
            }
        except Exception as e:
            return {
                "client_type": "azure_openai",
                "error": f"Metadata extraction failed: {e}",
                "timing": {
                    "request_start": request_start_time.isoformat(),
                    "response_end": request_end_time.isoformat(),
                    "duration_ms": int(
                        (request_end_time - request_start_time).total_seconds()
                        * 1000
                    ),
                },
            }

    def prompt_llm(
        self,
        model: str = None,
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
        if model is None:
            model = DEPLOYMENT_NAME

        attempts = 0
        last_error = None

        if message_history:
            messages = message_history
        elif question:
            content = [{"type": "text", "text": question}]
            if base64_images:
                for img in base64_images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            messages = [{"role": "user", "content": content}]
        else:
            raise ValueError("Either message_history or question must be provided")

        while attempts <= retries:
            try:
                request_start_time = datetime.now()

                completion_params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                if model not in REASONING_MODELS_LIST:
                    completion_params["logprobs"] = True
                    completion_params["top_logprobs"] = 5

                if model in REASONING_MODELS_LIST:
                    del completion_params["max_completion_tokens"]
                    del completion_params["temperature"]
                    del completion_params["top_p"]

                if response_model:
                    completion_params["response_format"] = {"type": "json_object"}

                completion = self.client.chat.completions.create(**completion_params)
                request_end_time = datetime.now()

                response_content = completion.choices[0].message.content

                metadata = self._extract_metadata(
                    completion,
                    request_start_time,
                    request_end_time,
                    messages,
                    temperature,
                    top_p,
                    max_tokens,
                    attempts,
                    model,
                )

                raw_response_length = len(response_content) if response_content else 0

                if response_model:
                    clean_json = self._extract_json_from_response(response_content)
                    validated_response, response_json = self._validate_response(
                        clean_json, response_model
                    )
                    metadata["processing"].update(
                        {
                            "json_parse_success": True,
                            "raw_response_length": raw_response_length,
                            "extracted_answer_length": len(str(validated_response)),
                        }
                    )
                    return {
                        "validated_response": validated_response,
                        "raw_json": response_json,
                        "metadata": metadata,
                    }

                extracted_answer = response_content
                json_parse_success = True
                try:
                    clean_json = self._extract_json_from_response(response_content)
                    response_json = json.loads(clean_json)
                    if isinstance(response_json, dict):
                        extracted_answer = response_json.get("answer", response_content)
                    else:
                        extracted_answer = response_content
                except (json.JSONDecodeError, TypeError):
                    json_parse_success = False

                metadata["processing"].update(
                    {
                        "json_parse_success": json_parse_success,
                        "raw_response_length": raw_response_length,
                        "extracted_answer_length": len(extracted_answer)
                        if extracted_answer
                        else 0,
                    }
                )

                return {"response": extracted_answer, "metadata": metadata}

            except Exception as e:
                print(e)
                last_error = e
                err_str = str(e)

                # Don't retry content filter errors — they'll always fail
                if "content_filter" in err_str or "ResponsibleAI" in err_str:
                    raise

                attempts += 1

                if attempts <= retries:
                    if "429" in err_str or "RateLimitReached" in err_str:
                        backoff = min(retry_delay * (2 ** (attempts - 1)), 120)
                    else:
                        backoff = retry_delay
                    print(
                        f"Retrying in {backoff}s... (attempt {attempts}/{retries})"
                    )
                    time.sleep(backoff)
                else:
                    print(f"All {retries} retries exhausted. Last error: {last_error}")
                    raise last_error

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except ImportError:
            return int(len(text.split()) * 1.3)
