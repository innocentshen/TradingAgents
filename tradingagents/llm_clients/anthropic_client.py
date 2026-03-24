from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
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


class RetryingChatAnthropic(ChatAnthropic):
    """ChatAnthropic with extra transport-level retries for transient disconnects."""

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
                return ChatAnthropic.invoke(retry_client, input, config, **kwargs)
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


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        llm_kwargs = {"model": self.model}

        for key in (
            "timeout",
            "max_retries",
            "api_key",
            "max_tokens",
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

        return RetryingChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
