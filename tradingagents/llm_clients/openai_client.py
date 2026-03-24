import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
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


class UnifiedChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips temperature/top_p for GPT-5 family models.

    GPT-5 family models use reasoning natively. temperature/top_p are only
    accepted when reasoning.effort is 'none'; with any other effort level
    (or for older GPT-5/GPT-5-mini/GPT-5-nano which always reason) the API
    rejects these params. Langchain defaults temperature=0.7, so we must
    strip it to avoid errors.

    Non-GPT-5 models (GPT-4.1, xAI, Ollama, etc.) are unaffected.
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

        if "gpt-5" in kwargs.get("model", "").lower():
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

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

    def _build_retry_client(self):
        return type(self)(**dict(self._client_init_kwargs))

    def invoke(self, input, config=None, **kwargs):
        last_error = None

        for attempt in range(1, self._transport_max_retries + 1):
            try:
                retry_client = self._build_retry_client()
                return ChatOpenAI.invoke(retry_client, input, config, **kwargs)
            except Exception as exc:
                last_error = exc
                if not is_retryable_transport_exception(exc):
                    raise
                if attempt >= self._transport_max_retries:
                    raise
                sleep_before_retry(
                    attempt,
                    self._transport_retry_backoff,
                    self._transport_retry_backoff_multiplier,
                    self._transport_retry_max_backoff,
                )

        raise last_error


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        llm_kwargs = {"model": self.model}

        if self.provider == "xai":
            llm_kwargs["base_url"] = "https://api.x.ai/v1"
            api_key = os.environ.get("XAI_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
        elif self.provider == "openrouter":
            llm_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
        elif self.provider == "ollama":
            llm_kwargs["base_url"] = "http://localhost:11434/v1"
            llm_kwargs["api_key"] = "ollama"  # Ollama doesn't require auth
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in (
            "timeout",
            "max_retries",
            "reasoning_effort",
            "api_key",
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
        llm_kwargs.setdefault("transport_retry_backoff", DEFAULT_TRANSPORT_RETRY_BACKOFF)
        llm_kwargs.setdefault(
            "transport_retry_backoff_multiplier",
            DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
        )
        llm_kwargs.setdefault(
            "transport_retry_max_backoff", DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF
        )

        return UnifiedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
