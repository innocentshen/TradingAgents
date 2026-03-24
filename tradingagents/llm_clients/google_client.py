import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr

from .base_client import BaseLLMClient
from .transport_retry import (
    DEFAULT_TRANSPORT_MAX_RETRIES,
    DEFAULT_TRANSPORT_RETRY_BACKOFF,
    DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF,
    is_retryable_transport_exception,
    sleep_before_retry,
)
from .validators import validate_model


def _unique_non_empty(values: list[Optional[str]]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in deduped:
            deduped.append(value)
    return deduped


def _resolve_google_api_key_candidates(explicit_key: Optional[str] = None) -> list[str]:
    return _unique_non_empty(
        [
            explicit_key,
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
        ]
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list: [{'type': 'text', 'text': '...'}]
    This normalizes to string for consistent downstream handling.
    """

    _transport_max_retries: int = PrivateAttr(default=DEFAULT_TRANSPORT_MAX_RETRIES)
    _transport_retry_backoff: float = PrivateAttr(default=DEFAULT_TRANSPORT_RETRY_BACKOFF)
    _transport_retry_backoff_multiplier: float = PrivateAttr(
        default=DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER
    )
    _transport_retry_max_backoff: float = PrivateAttr(
        default=DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF
    )
    _client_init_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _api_key_candidates: list[Optional[str]] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        transport_max_retries = max(
            1, int(kwargs.pop("transport_max_retries", DEFAULT_TRANSPORT_MAX_RETRIES))
        )
        transport_retry_backoff = max(
            0.0,
            float(kwargs.pop("transport_retry_backoff", DEFAULT_TRANSPORT_RETRY_BACKOFF)),
        )
        transport_retry_backoff_multiplier = max(
            1.0,
            float(
                kwargs.pop(
                    "transport_retry_backoff_multiplier",
                    DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
                )
            ),
        )
        transport_retry_max_backoff = max(
            0.0,
            float(
                kwargs.pop(
                    "transport_retry_max_backoff",
                    DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF,
                )
            ),
        )
        api_key_candidates = kwargs.pop("google_api_key_candidates", None)
        client_init_kwargs = dict(kwargs)
        client_init_kwargs.update(
            {
                "transport_max_retries": transport_max_retries,
                "transport_retry_backoff": transport_retry_backoff,
                "transport_retry_backoff_multiplier": transport_retry_backoff_multiplier,
                "transport_retry_max_backoff": transport_retry_max_backoff,
            }
        )
        super().__init__(**kwargs)
        self._transport_max_retries = transport_max_retries
        self._transport_retry_backoff = transport_retry_backoff
        self._transport_retry_backoff_multiplier = transport_retry_backoff_multiplier
        self._transport_retry_max_backoff = transport_retry_max_backoff
        self._client_init_kwargs = client_init_kwargs
        self._api_key_candidates = list(api_key_candidates or [kwargs.get("google_api_key")])

    def _normalize_content(self, response):
        content = response.content
        if isinstance(content, list):
            texts = [
                item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else item if isinstance(item, str) else ""
                for item in content
            ]
            response.content = "\n".join(t for t in texts if t)
        return response

    def _build_retry_client(self, api_key: Optional[str]):
        client_kwargs = dict(self._client_init_kwargs)
        if api_key:
            client_kwargs["google_api_key"] = api_key
        else:
            client_kwargs.pop("google_api_key", None)
        return type(self)(**client_kwargs)

    def invoke(self, input, config=None, **kwargs):
        last_error = None

        for api_key_index, api_key in enumerate(self._api_key_candidates):
            for attempt in range(1, self._transport_max_retries + 1):
                try:
                    retry_client = self._build_retry_client(api_key)
                    response = ChatGoogleGenerativeAI.invoke(
                        retry_client, input, config, **kwargs
                    )
                    return self._normalize_content(response)
                except Exception as exc:
                    last_error = exc
                    if not is_retryable_transport_exception(exc):
                        raise

                    has_more_attempts = attempt < self._transport_max_retries
                    has_more_api_keys = api_key_index < len(self._api_key_candidates) - 1
                    if not (has_more_attempts or has_more_api_keys):
                        raise

                    sleep_before_retry(
                        attempt,
                        self._transport_retry_backoff,
                        self._transport_retry_backoff_multiplier,
                        self._transport_retry_max_backoff,
                    )

        raise last_error


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        llm_kwargs = {"model": self.model}
        api_key_candidates = _resolve_google_api_key_candidates(
            self.kwargs.get("google_api_key")
        )

        for key in (
            "timeout",
            "max_retries",
            "google_api_key",
            "callbacks",
            "http_client",
            "http_async_client",
            "transport_max_retries",
            "transport_retry_backoff",
            "transport_retry_backoff_multiplier",
            "transport_retry_max_backoff",
        ):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        llm_kwargs.setdefault("transport_max_retries", DEFAULT_TRANSPORT_MAX_RETRIES)
        llm_kwargs.setdefault(
            "transport_retry_backoff", DEFAULT_TRANSPORT_RETRY_BACKOFF
        )
        llm_kwargs.setdefault(
            "transport_retry_backoff_multiplier",
            DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
        )
        llm_kwargs.setdefault(
            "transport_retry_max_backoff", DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF
        )

        if api_key_candidates:
            llm_kwargs["google_api_key"] = api_key_candidates[0]
            llm_kwargs["google_api_key_candidates"] = api_key_candidates

        # Map thinking_level to appropriate API param based on model
        # Gemini 3 Pro: low, high
        # Gemini 3 Flash: minimal, low, medium, high
        # Gemini 2.5: thinking_budget (0=disable, -1=dynamic)
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
