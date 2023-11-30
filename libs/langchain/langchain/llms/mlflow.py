from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, PrivateAttr


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"
class Params(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for MLflow"""

    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class Mlflow(LLM):
    """Wrapper around completions LLMs in MLflow.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/server.html.

    Example:
        .. code-block:: python

            from langchain.llms import Mlflow

            completions = Mlflow(
                target_uri="http://localhost:5000",
                endpoint="test",
                params={"temperature": 0.1}
            )
    """

    endpoint: str
    """The endpoint to use."""
    target_uri: str
    """The target URI to use."""
    params: Optional[Params] = None
    """Extra parameters such as `temperature`."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                "Please run `pip install mlflow[genai]` to install "
                "required dependencies."
            ) from e

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "target_uri": self.target_uri,
            "endpoint": self.endpoint,
        }
        if self.params:
            params["params"] = self.params.dict()
        return params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        data: Dict[str, Any] = {
            "prompt": prompt,
            **(self.params.dict() if self.params else {}),
        }
        if s := (stop or (self.params.stop if self.params else None)):
            data["stop"] = s
        resp = self._client.predict(endpoint=self.endpoint, inputs=data)
        return resp["choices"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "mlflow"
