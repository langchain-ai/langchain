import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

__all__ = ["Databricks"]


class _DatabricksClientBase(BaseModel, ABC):
    """A base JSON API client that talks to Databricks."""

    api_url: str
    api_token: str

    def request(self, method: str, url: str, request: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.request(
            method=method, url=url, headers=headers, json=request
        )
        # TODO: error handling and automatic retries
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")
        return response.json()

    def _get(self, url: str) -> Any:
        return self.request("GET", url, None)

    def _post(self, url: str, request: Any) -> Any:
        return self.request("POST", url, request)

    @abstractmethod
    def post(
        self, request: Any, transform_output_fn: Optional[Callable[..., str]] = None
    ) -> Any: ...

    @property
    def llm(self) -> bool:
        return False


def _transform_completions(response: Dict[str, Any]) -> str:
    return response["choices"][0]["text"]


def _transform_llama2_chat(response: Dict[str, Any]) -> str:
    return response["candidates"][0]["text"]


def _transform_chat(response: Dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


class _DatabricksServingEndpointClient(_DatabricksClientBase):
    """An API client that talks to a Databricks serving endpoint."""

    host: str
    endpoint_name: str
    databricks_uri: str
    client: Any = None
    external_or_foundation: bool = False
    task: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)

        try:
            from mlflow.deployments import get_deploy_client

            self.client = get_deploy_client(self.databricks_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                "Please install mlflow with `pip install mlflow`."
            ) from e

        endpoint = self.client.get_endpoint(self.endpoint_name)
        self.external_or_foundation = endpoint.get("endpoint_type", "").lower() in (
            "external_model",
            "foundation_model_api",
        )
        if self.task is None:
            self.task = endpoint.get("task")

    @property
    def llm(self) -> bool:
        return self.task in ("llm/v1/chat", "llm/v1/completions", "llama2/chat")

    @model_validator(mode="before")
    @classmethod
    def set_api_url(cls, values: Dict[str, Any]) -> Any:
        if "api_url" not in values:
            host = values["host"]
            endpoint_name = values["endpoint_name"]
            api_url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
            values["api_url"] = api_url
        return values

    def post(
        self, request: Any, transform_output_fn: Optional[Callable[..., str]] = None
    ) -> Any:
        if self.external_or_foundation:
            resp = self.client.predict(endpoint=self.endpoint_name, inputs=request)
            if transform_output_fn:
                return transform_output_fn(resp)

            if self.task == "llm/v1/chat":
                return _transform_chat(resp)
            elif self.task == "llm/v1/completions":
                return _transform_completions(resp)

            return resp
        else:
            # See https://docs.databricks.com/machine-learning/model-serving/score-model-serving-endpoints.html
            wrapped_request = {"dataframe_records": [request]}
            response = self.client.predict(
                endpoint=self.endpoint_name, inputs=wrapped_request
            )
            preds = response["predictions"]
            # For a single-record query, the result is not a list.
            pred = preds[0] if isinstance(preds, list) else preds
            if self.task == "llama2/chat":
                return _transform_llama2_chat(pred)
            return transform_output_fn(pred) if transform_output_fn else pred


class _DatabricksClusterDriverProxyClient(_DatabricksClientBase):
    """An API client that talks to a Databricks cluster driver proxy app."""

    host: str
    cluster_id: str
    cluster_driver_port: str

    @model_validator(mode="before")
    @classmethod
    def set_api_url(cls, values: Dict[str, Any]) -> Any:
        if "api_url" not in values:
            host = values["host"]
            cluster_id = values["cluster_id"]
            port = values["cluster_driver_port"]
            api_url = f"https://{host}/driver-proxy-api/o/0/{cluster_id}/{port}"
            values["api_url"] = api_url
        return values

    def post(
        self, request: Any, transform_output_fn: Optional[Callable[..., str]] = None
    ) -> Any:
        resp = self._post(self.api_url, request)
        return transform_output_fn(resp) if transform_output_fn else resp


def get_repl_context() -> Any:
    """Get the notebook REPL context if running inside a Databricks notebook.
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
    """Get the default Databricks workspace hostname.
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
    """Get the default Databricks personal access token.
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


def _is_hex_string(data: str) -> bool:
    """Checks if a data is a valid hexadecimal string using a regular expression."""
    if not isinstance(data, str):
        return False
    pattern = r"^[0-9a-fA-F]+$"
    return bool(re.match(pattern, data))


def _load_pickled_fn_from_hex_string(
    data: str, allow_dangerous_deserialization: Optional[bool]
) -> Callable:
    """Loads a pickled function from a hexadecimal string."""
    if not allow_dangerous_deserialization:
        raise ValueError(
            "This code relies on the pickle module. "
            "You will need to set allow_dangerous_deserialization=True "
            "if you want to opt-in to allow deserialization of data using pickle."
            "Data can be compromised by a malicious actor if "
            "not handled properly to include "
            "a malicious payload that when deserialized with "
            "pickle can execute arbitrary code on your machine."
        )

    try:
        import cloudpickle
    except Exception as e:
        raise ValueError(f"Please install cloudpickle>=2.0.0. Error: {e}")

    try:
        return cloudpickle.loads(bytes.fromhex(data))  # ignore[pickle]: explicit-opt-in
    except Exception as e:
        raise ValueError(
            f"Failed to load the pickled function from a hexadecimal string. Error: {e}"
        )


def _pickle_fn_to_hex_string(fn: Callable) -> str:
    """Pickles a function and returns the hexadecimal string."""
    try:
        import cloudpickle
    except Exception as e:
        raise ValueError(f"Please install cloudpickle>=2.0.0. Error: {e}")

    try:
        return cloudpickle.dumps(fn).hex()
    except Exception as e:
        raise ValueError(f"Failed to pickle the function: {e}")


@deprecated(
    since="0.3.3",
    removal="1.0",
    alternative_import="databricks_langchain.ChatDatabricks",
)
class Databricks(LLM):
    """Databricks serving endpoint or a cluster driver proxy app for LLM.

    It supports two endpoint types:

    * **Serving endpoint** (recommended for both production and development).
      We assume that an LLM was deployed to a serving endpoint.
      To wrap it as an LLM you must have "Can Query" permission to the endpoint.
      Set ``endpoint_name`` accordingly and do not set ``cluster_id`` and
      ``cluster_driver_port``.

      If the underlying model is a model registered by MLflow, the expected model
      signature is:

      * inputs::

          [{"name": "prompt", "type": "string"},
           {"name": "stop", "type": "list[string]"}]

      * outputs: ``[{"type": "string"}]``

      If the underlying model is an external or foundation model, the response from the
      endpoint is automatically transformed to the expected format unless
      ``transform_output_fn`` is provided.

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
    """
    Deprecated. Please use ``extra_params`` instead. Extra parameters to pass to
    the endpoint.
    """

    transform_input_fn: Optional[Callable] = None
    """A function that transforms ``{prompt, stop, **kwargs}`` into a JSON-compatible
    request object that the endpoint accepts.
    For example, you can apply a prompt template to the input prompt.
    """

    transform_output_fn: Optional[Callable[..., str]] = None
    """A function that transforms the output from the endpoint to the generated text.
    """

    databricks_uri: str = "databricks"
    """The databricks URI. Only used when using a serving endpoint."""

    temperature: float = 0.0
    """The sampling temperature."""
    n: int = 1
    """The number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """The stop sequence."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    """Any extra parameters to pass to the endpoint."""
    task: Optional[str] = None
    """The task of the endpoint. Only used when using a serving endpoint.
    If not provided, the task is automatically inferred from the endpoint.
    """

    allow_dangerous_deserialization: bool = False
    """Whether to allow dangerous deserialization of the data which 
    involves loading data using pickle.
    
    If the data has been modified by a malicious actor, it can deliver a
    malicious payload that results in execution of arbitrary code on the target
    machine.
    """

    _client: _DatabricksClientBase = PrivateAttr()

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def _llm_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": self.temperature,
            "n": self.n,
        }
        if self.stop:
            params["stop"] = self.stop
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    @model_validator(mode="before")
    @classmethod
    def set_cluster_id(cls, values: Dict[str, Any]) -> dict:
        cluster_id = values.get("cluster_id")
        endpoint_name = values.get("endpoint_name")
        if cluster_id and endpoint_name:
            raise ValueError("Cannot set both endpoint_name and cluster_id.")
        elif endpoint_name:
            values["cluster_id"] = None
        elif cluster_id:
            pass
        else:
            try:
                if context_cluster_id := get_repl_context().clusterId:
                    values["cluster_id"] = context_cluster_id
                raise ValueError("Context doesn't contain clusterId.")
            except Exception as e:
                raise ValueError(
                    "Neither endpoint_name nor cluster_id was set. "
                    "And the cluster_id cannot be automatically determined. Received"
                    f" error: {e}"
                )

        cluster_driver_port = values.get("cluster_driver_port")
        if cluster_driver_port and endpoint_name:
            raise ValueError("Cannot set both endpoint_name and cluster_driver_port.")
        elif endpoint_name:
            values["cluster_driver_port"] = None
        elif cluster_driver_port is None:
            raise ValueError(
                "Must set cluster_driver_port to connect to a cluster driver."
            )
        elif int(cluster_driver_port) <= 0:
            raise ValueError(f"Invalid cluster_driver_port: {cluster_driver_port}")
        else:
            pass

        if model_kwargs := values.get("model_kwargs"):
            assert (
                "prompt" not in model_kwargs
            ), "model_kwargs must not contain key 'prompt'"
            assert (
                "stop" not in model_kwargs
            ), "model_kwargs must not contain key 'stop'"
        return values

    def __init__(self, **data: Any):
        if "transform_input_fn" in data and _is_hex_string(data["transform_input_fn"]):
            data["transform_input_fn"] = _load_pickled_fn_from_hex_string(
                data=data["transform_input_fn"],
                allow_dangerous_deserialization=data.get(
                    "allow_dangerous_deserialization"
                ),
            )
        if "transform_output_fn" in data and _is_hex_string(
            data["transform_output_fn"]
        ):
            data["transform_output_fn"] = _load_pickled_fn_from_hex_string(
                data=data["transform_output_fn"],
                allow_dangerous_deserialization=data.get(
                    "allow_dangerous_deserialization"
                ),
            )

        super().__init__(**data)
        if self.model_kwargs is not None and self.extra_params is not None:
            raise ValueError("Cannot set both extra_params and extra_params.")
        elif self.model_kwargs is not None:
            warnings.warn(
                "model_kwargs is deprecated. Please use extra_params instead.",
                DeprecationWarning,
            )
        if self.endpoint_name:
            self._client = _DatabricksServingEndpointClient(
                host=self.host,
                api_token=self.api_token,
                endpoint_name=self.endpoint_name,
                databricks_uri=self.databricks_uri,
                task=self.task,
            )
        elif self.cluster_id and self.cluster_driver_port:
            self._client = _DatabricksClusterDriverProxyClient(  # type: ignore[call-arg]
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
    def _default_params(self) -> Dict[str, Any]:
        """Return default params."""
        return {
            "host": self.host,
            # "api_token": self.api_token,  # Never save the token
            "endpoint_name": self.endpoint_name,
            "cluster_id": self.cluster_id,
            "cluster_driver_port": self.cluster_driver_port,
            "databricks_uri": self.databricks_uri,
            "model_kwargs": self.model_kwargs,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "extra_params": self.extra_params,
            "task": self.task,
            "transform_input_fn": None
            if self.transform_input_fn is None
            else _pickle_fn_to_hex_string(self.transform_input_fn),
            "transform_output_fn": None
            if self.transform_output_fn is None
            else _pickle_fn_to_hex_string(self.transform_output_fn),
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params

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

        request: Dict[str, Any] = {"prompt": prompt}
        if self._client.llm:
            request.update(self._llm_params)
        request.update(self.model_kwargs or self.extra_params)
        request.update(kwargs)
        if stop:
            request["stop"] = stop

        if self.transform_input_fn:
            request = self.transform_input_fn(**request)

        return self._client.post(request, transform_output_fn=self.transform_output_fn)
