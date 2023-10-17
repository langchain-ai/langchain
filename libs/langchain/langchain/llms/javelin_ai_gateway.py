from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class Params(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for the Javelin AI Gateway LLM."""

    temperature: float = 0.0
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class JavelinAIGateway(LLM):
    """
    Wrapper around completions LLMs in the Javelin AI Gateway.

    To use, you should have the ``javelin_sdk`` python package installed.
    For more information, see https://docs.getjavelin.io

    Example:
        .. code-block:: python

            from langchain.llms import JavelinAIGateway

            completions = JavelinAIGateway(
                gateway_uri="<your-javelin-ai-gateway-uri>",
                route="<your-javelin-ai-gateway-completions-route>",
                params={
                    "temperature": 0.1
                }
            )
    """

    route: str
    """The route to use for the Javelin AI Gateway API."""

    client: Optional[Any] = None
    """The Javelin AI Gateway client."""

    gateway_uri: Optional[str] = None
    """The URI of the Javelin AI Gateway API."""

    params: Optional[Params] = None
    """Parameters for the Javelin AI Gateway API."""

    javelin_api_key: Optional[str] = None
    """The API key for the Javelin AI Gateway API."""

    def __init__(self, **kwargs: Any):
        try:
            from javelin_sdk import (
                JavelinClient,
                UnauthorizedError,
            )
        except ImportError:
            raise ImportError(
                "Could not import javelin_sdk python package. "
                "Please install it with `pip install javelin_sdk`."
            )
        super().__init__(**kwargs)
        if self.gateway_uri:
            try:
                self.client = JavelinClient(
                    base_url=self.gateway_uri, api_key=self.javelin_api_key
                )
            except UnauthorizedError as e:
                raise ValueError("Javelin: Incorrect API Key.") from e

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Javelin AI Gateway API."""
        params: Dict[str, Any] = {
            "gateway_uri": self.gateway_uri,
            "route": self.route,
            "javelin_api_key": self.javelin_api_key,
            **(self.params.dict() if self.params else {}),
        }
        return params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Javelin AI Gateway API."""
        data: Dict[str, Any] = {
            "prompt": prompt,
            **(self.params.dict() if self.params else {}),
        }
        if s := (stop or (self.params.stop if self.params else None)):
            data["stop"] = s

        if self.client is not None:
            resp = self.client.query_route(self.route, query_body=data)
        else:
            raise ValueError("Javelin client is not initialized.")

        resp_dict = resp.dict()

        try:
            return resp_dict["llm_response"]["choices"][0]["text"]
        except KeyError:
            return ""

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call async the Javelin AI Gateway API."""
        data: Dict[str, Any] = {
            "prompt": prompt,
            **(self.params.dict() if self.params else {}),
        }
        if s := (stop or (self.params.stop if self.params else None)):
            data["stop"] = s

        if self.client is not None:
            resp = await self.client.aquery_route(self.route, query_body=data)
        else:
            raise ValueError("Javelin client is not initialized.")

        resp_dict = resp.dict()

        try:
            return resp_dict["llm_response"]["choices"][0]["text"]
        except KeyError:
            return ""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "javelin-ai-gateway"
