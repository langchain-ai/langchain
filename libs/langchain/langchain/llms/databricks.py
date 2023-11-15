import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)

__all__ = ["Databricks"]


class _DatabricksClientBase(BaseModel, ABC):
    """A base JSON API client that talks to Databricks."""

    api_url: str
    api_token: str

    def post_raw(self, request: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.post(self.api_url, headers=headers, json=request)
        # TODO: error handling and automatic retries
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")
        return response.json()

    @abstractmethod
    def post(self, request: Any) -> Any:
        ...


class _DatabricksServingEndpointClient(_DatabricksClientBase):
    """An API client that talks to a Databricks serving endpoint."""

    host: str
    endpoint_name: str

    @root_validator(pre=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "api_url" not in values:
            host = values["host"]
            endpoint_name = values["endpoint_name"]
            api_url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
            values["api_url"] = api_url
        return values

    def post(self, request: Any) -> Any:
        # See https://docs.databricks.com/machine-learning/model-serving/score-model-serving-endpoints.html
        wrapped_request = {"dataframe_records": [request]}
        response = self.post_raw(wrapped_request)["predictions"]
        # For a single-record query, the result is not a list.
        if isinstance(response, list):
            response = response[0]
        return response


class _DatabricksClusterDriverProxyClient(_DatabricksClientBase):
    """An API client that talks to a Databricks cluster driver proxy app."""

    host: str
    cluster_id: str
    cluster_driver_port: str

    @root_validator(pre=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "api_url" not in values:
            host = values["host"]
            cluster_id = values["cluster_id"]
            port = values["cluster_driver_port"]
            api_url = f"https://{host}/driver-proxy-api/o/0/{cluster_id}/{port}"
            values["api_url"] = api_url
        return values

    def post(self, request: Any) -> Any:
        return self.post_raw(request)


def get_repl_context() -> Any:
    """Gets the notebook REPL context if running inside a Databricks notebook.
    Returns None otherwise.
    """
    try:
        from dbruntime.databricks_repl_context import get_context

        return get_context()
    except ImportError:
        raise ImportError(
            "Cannot access dbruntime, not running inside a Databricks notebook."
        )


def get_default_host() -> str:
    """Gets the default Databricks workspace hostname.
    Raises an error if the hostname cannot be automatically determined.
    """
    host = os.getenv("DATABRICKS_HOST")
    if not host:
        try:
            host = get_repl_context().browserHostName
            if not host:
                raise ValueError("context doesn't contain browserHostName.")
        except Exception as e:
            raise ValueError(
                "host was not set and cannot be automatically inferred. Set "
                f"environment variable 'DATABRICKS_HOST'. Received error: {e}"
            )
    # TODO: support Databricks CLI profile
    host = host.lstrip("https://").lstrip("http://").rstrip("/")
    return host


def get_default_api_token() -> str:
    """Gets the default Databricks personal access token.
    Raises an error if the token cannot be automatically determined.
    """
    if api_token := os.getenv("DATABRICKS_TOKEN"):
        return api_token
    try:
        api_token = get_repl_context().apiToken
        if not api_token:
            raise ValueError("context doesn't contain apiToken.")
    except Exception as e:
        raise ValueError(
            "api_token was not set and cannot be automatically inferred. Set "
            f"environment variable 'DATABRICKS_TOKEN'. Received error: {e}"
        )
    # TODO: support Databricks CLI profile
    return api_token


class Databricks(LLM):
    """Databricks serving endpoint or a cluster driver proxy app for LLM.

    It supports two endpoint types:

    * **Serving endpoint** (recommended for both production and development).
      We assume that an LLM was registered and deployed to a serving endpoint.
      To wrap it as an LLM you must have "Can Query" permission to the endpoint.
      Set ``endpoint_name`` accordingly and do not set ``cluster_id`` and
      ``cluster_driver_port``.
      The expected model signature is:

      * inputs::

          [{"name": "prompt", "type": "string"},
           {"name": "stop", "type": "list[string]"}]

      * outputs: ``[{"type": "string"}]``

    * **Cluster driver proxy app** (recommended for interactive development).
      One can load an LLM on a Databricks interactive cluster and start a local HTTP
      server on the driver node to serve the model at ``/`` using HTTP POST method
      with JSON input/output.
      Please use a port number between ``[3000, 8000]`` and let the server listen to
      the driver IP address or simply ``0.0.0.0`` instead of localhost only.
      To wrap it as an LLM you must have "Can Attach To" permission to the cluster.
      Set ``cluster_id`` and ``cluster_driver_port`` and do not set ``endpoint_name``.
      The expected server schema (using JSON schema) is:

      * inputs::

          {"type": "object",
           "properties": {
              "prompt": {"type": "string"},
              "stop": {"type": "array", "items": {"type": "string"}}},
           "required": ["prompt"]}`

      * outputs: ``{"type": "string"}``

    If the endpoint model signature is different or you want to set extra params,
    you can use `transform_input_fn` and `transform_output_fn` to apply necessary
    transformations before and after the query.
    """

    host: str = Field(default_factory=get_default_host)
    """Databricks workspace hostname.
    If not provided, the default value is determined by

    * the ``DATABRICKS_HOST`` environment variable if present, or
    * the hostname of the current Databricks workspace if running inside
      a Databricks notebook attached to an interactive cluster in "single user"
      or "no isolation shared" mode.
    """

    api_token: str = Field(default_factory=get_default_api_token)
    """Databricks personal access token.
    If not provided, the default value is determined by

    * the ``DATABRICKS_TOKEN`` environment variable if present, or
    * an automatically generated temporary token if running inside a Databricks
      notebook attached to an interactive cluster in "single user" or
      "no isolation shared" mode.
    """

    endpoint_name: Optional[str] = None
    """Name of the model serving endpoint.
    You must specify the endpoint name to connect to a model serving endpoint.
    You must not set both ``endpoint_name`` and ``cluster_id``.
    """

    cluster_id: Optional[str] = None
    """ID of the cluster if connecting to a cluster driver proxy app.
    If neither ``endpoint_name`` nor ``cluster_id`` is not provided and the code runs
    inside a Databricks notebook attached to an interactive cluster in "single user"
    or "no isolation shared" mode, the current cluster ID is used as default.
    You must not set both ``endpoint_name`` and ``cluster_id``.
    """

    cluster_driver_port: Optional[str] = None
    """The port number used by the HTTP server running on the cluster driver node.
    The server should listen on the driver IP address or simply ``0.0.0.0`` to connect.
    We recommend the server using a port number between ``[3000, 8000]``.
    """

    model_kwargs: Optional[Dict[str, Any]] = None
    """Extra parameters to pass to the endpoint."""

    transform_input_fn: Optional[Callable] = None
    """A function that transforms ``{prompt, stop, **kwargs}`` into a JSON-compatible
    request object that the endpoint accepts.
    For example, you can apply a prompt template to the input prompt.
    """

    transform_output_fn: Optional[Callable[..., str]] = None
    """A function that transforms the output from the endpoint to the generated text.
    """

    _client: _DatabricksClientBase = PrivateAttr()

    class Config:
        extra = Extra.forbid
        underscore_attrs_are_private = True

    @validator("cluster_id", always=True)
    def set_cluster_id(cls, v: Any, values: Dict[str, Any]) -> Optional[str]:
        if v and values["endpoint_name"]:
            raise ValueError("Cannot set both endpoint_name and cluster_id.")
        elif values["endpoint_name"]:
            return None
        elif v:
            return v
        else:
            try:
                if v := get_repl_context().clusterId:
                    return v
                raise ValueError("Context doesn't contain clusterId.")
            except Exception as e:
                raise ValueError(
                    "Neither endpoint_name nor cluster_id was set. "
                    "And the cluster_id cannot be automatically determined. Received"
                    f" error: {e}"
                )

    @validator("cluster_driver_port", always=True)
    def set_cluster_driver_port(cls, v: Any, values: Dict[str, Any]) -> Optional[str]:
        if v and values["endpoint_name"]:
            raise ValueError("Cannot set both endpoint_name and cluster_driver_port.")
        elif values["endpoint_name"]:
            return None
        elif v is None:
            raise ValueError(
                "Must set cluster_driver_port to connect to a cluster driver."
            )
        elif int(v) <= 0:
            raise ValueError(f"Invalid cluster_driver_port: {v}")
        else:
            return v

    @validator("model_kwargs", always=True)
    def set_model_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v:
            assert "prompt" not in v, "model_kwargs must not contain key 'prompt'"
            assert "stop" not in v, "model_kwargs must not contain key 'stop'"
        return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.endpoint_name:
            self._client = _DatabricksServingEndpointClient(
                host=self.host,
                api_token=self.api_token,
                endpoint_name=self.endpoint_name,
            )
        elif self.cluster_id and self.cluster_driver_port:
            self._client = _DatabricksClusterDriverProxyClient(
                host=self.host,
                api_token=self.api_token,
                cluster_id=self.cluster_id,
                cluster_driver_port=self.cluster_driver_port,
            )
        else:
            raise ValueError(
                "Must specify either endpoint_name or cluster_id/cluster_driver_port."
            )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "databricks"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Queries the LLM endpoint with the given prompt and stop sequence."""

        # TODO: support callbacks

        request = {"prompt": prompt, "stop": stop}
        request.update(kwargs)
        if self.model_kwargs:
            request.update(self.model_kwargs)

        if self.transform_input_fn:
            request = self.transform_input_fn(**request)

        response = self._client.post(request)

        if self.transform_output_fn:
            response = self.transform_output_fn(response)

        return response
