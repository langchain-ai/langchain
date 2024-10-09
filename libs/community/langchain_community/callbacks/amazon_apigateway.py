from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class StreamingAmazonAPIGatewayWebSocketCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(
        self,
        boto3_session: Any,
        endpoint_url: str,
        connection_id: str,
        *,
        on_token: Callable[[str], str] | None = None,
        on_end: Callable[[], str] | None = None,
        on_err: Callable[[BaseException], str] | None = None,
    ) -> None:
        """
        Initialize callback handler
        with boto3 session and api gateway websocket event.
        """
        self.connection_id = connection_id
        self.apigw = boto3_session.client(
            "apigatewaymanagementapi",
            endpoint_url=endpoint_url,
        )
        self.on_token = on_token
        self.on_end = on_end
        self.on_err = on_err
        super().__init__()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.on_token is not None:
            self.apigw.post_to_connection(
                ConnectionId=self.connection_id,
                Data=self.on_token(token),
            )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        if self.on_end is not None:
            self.apigw.post_to_connection(
                ConnectionId=self.connection_id,
                Data=self.on_end(),
            )

    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> Any:
        if self.on_err is not None:
            self.apigw.post_to_connection(
                ConnectionId=self.connection_id,
                Data=self.on_err(error),
            )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
