from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class Params(BaseModel, extra=Extra.allow):
    """Parameters for the MLflow AI Gateway LLM."""

    temperature: float = 0.0
    candidate_count: int = 1
    """The number of candidates to return."""
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class MlflowAIGateway(LLM):
    """The MLflow AI Gateway models."""

    route: str
    gateway_uri: Optional[str] = None
    params: Optional[Params] = None

    def __init__(self, **kwargs: Any):
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        super().__init__(**kwargs)
        if self.gateway_uri:
            mlflow.gateway.set_gateway_uri(self.gateway_uri)

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "gateway_uri": self.gateway_uri,
            "route": self.route,
            **(self.params.dict() if self.params else {}),
        }
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
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        data: Dict[str, Any] = {
            "prompt": prompt,
            **(self.params.dict() if self.params else {}),
        }
        if s := (stop or (self.params.stop if self.params else None)):
            data["stop"] = s
        resp = mlflow.gateway.query(self.route, data=data)
        return resp["candidates"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "mlflow-ai-gateway"
