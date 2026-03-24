import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from tradingagents.llm_clients.openai_client import UnifiedChatOpenAI


class OpenAIRetryTests(unittest.TestCase):
    @patch("tradingagents.llm_clients.openai_client.sleep_before_retry")
    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.invoke")
    def test_invoke_retries_transient_transport_disconnects(
        self,
        mock_invoke,
        mock_sleep,
    ):
        responses = [
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            SimpleNamespace(content="ok"),
        ]

        def side_effect(*args, **kwargs):
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        mock_invoke.side_effect = side_effect

        llm = UnifiedChatOpenAI(
            model="gpt-5-mini",
            api_key="test-key",
            base_url="http://example.com/v1",
            transport_max_retries=2,
        )

        result = llm.invoke("hello")

        self.assertEqual(result.content, "ok")
        self.assertEqual(mock_invoke.call_count, 2)
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
