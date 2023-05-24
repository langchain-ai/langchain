from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Union, Callable

from pydantic import BaseModel, Extra, Field, validator, root_validator, ValidationError, PrivateAttr
import requests

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

__all__ = ["DatabricksEndpoint"]


class DatabricksClientBase(BaseModel, ABC):
    api_url: str
    api_token: str

    def post_raw(self, request):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.post(self.api_url, headers=headers, json=request)
        # TODO: error handling and automatic retries
        assert response.ok, f"HTTP {response.status_code} error: {response.text}"
        return response.json()
    
    @abstractmethod
    def post(self, request):
        raise NotImplementedError


class DatabricksServingEndpointClient(DatabricksClientBase):
    host: str
    endpoint_name: str

    @root_validator(pre=True)
    def set_api_url(cls, values):
        if "api_url" not in values:
            values["api_url"] = \
                f"https://{values['host']}/serving-endpoints/{values['endpoint_name']}/invocations"
        return values

    def post(self, request):
        # See https://docs.databricks.com/machine-learning/model-serving/score-model-serving-endpoints.html
        wrapped_request = {"dataframe_records": [request]}
        response = self.post_raw(wrapped_request)["predictions"]
        # For a signle-record query, the result is not a list.
        if isinstance(response, list):
            response = response[0]
        return response

class DatabricksClusterDriverProxyClient(DatabricksClientBase):
    host: str
    cluster_id: str
    cluster_driver_port: str

    @root_validator(pre=True)
    def set_api_url(cls, values):
        if "api_url" not in values:
            values["api_url"] = \
                f"https://{values['host']}/driver-proxy-api/o/0/{values['cluster_id']}/{values['cluster_driver_port']}"
        return values

    def post(self, request):
        return self.post_raw(request)

def get_repl_context():
    """Gets the notebook REPL context if running inside a Databricks notebook.
    Returns None otherwise.
    """
    context = None
    try:
        from dbruntime.databricks_repl_context import get_context
        context = get_context()
    except ImportError:
        logging.debug("Not running inside a Databricks notebook")
    except Exception:
        logging.debug(
            "Error getting the REPL context. Will not use it to provide default values.",
            exc_info=True)
    return context

def get_default_host():
    host = os.getenv("DATABRICKS_HOST")
    if not host:
        try:
            host = get_repl_context().browserHostName
        except:
            pass
    assert host, "host was not set and it cannot be automatically determined."
    return host

def get_default_api_token():
    api_token = os.getenv("DATABRIKCS_API_TOKEN")
    if not api_token:
        try:
            api_token = get_repl_context().apiToken
        except:
            pass
    assert api_token, "api_token was not set and it cannot be automatically determined."
    return api_token

class DatabricksEndpoint(LLM):
    """LLM wrapper around a Databricks endpoint. serving endpoint or a cluster driver proxy app.
    It supports two types of endpoints:
    - Serving endpoint (recommended for both development and production).
      We assume that an LLM was registered and deployed to a serving endpoint.
      To wrap it as a LangChain LLM you must have "Can Query" permission to the endpoint.
      Set `endpoint_name` accordingly and do not set `cluster_id` and `cluster_driver_port`.
      The expected model signature is:
      - inputs: `[{"name": "prompt", "type": "string"}, {"name": "stop", "type": "list[string]"}]`
      - outputs: `[{"type": "string"}]`

    - Cluster driver proxy app (recommended for rapid development).
      One can load an LLM on a Databricks interactive cluster and start a local HTTP server
      on the driver node to serve the model at "/" using HTTP POST with JSON input/output.
      Please use a port number between [3000, 8000] and let the server listen to the driver IP
      address or simply `0.0.0.0` instead of localhost only.
      To wrap it as a LangChain LLM you must have "Can Attach To" permission to the cluster.
      Set `cluster_id` and `cluster_driver_port` accordingly and do not set `endpoint_name`.
      The expected server schema (using JSON schema) is:
      - inputs: `{"type":"object","properties":{"prompt":{"type":"string"},"stop":{"type":"array","items":{"type":"string"}}},"required":["prompt"]}`
      - outputs: `{"type":"string"}`

    If the endpoint model signature is different or you want to set extra params,
    you can use `transform_input_fn` and `transform_output_fn` to apply necessary transformations
    before and after the query.
    """

    host: str = Field(default_factory=get_default_host)
    """Databricks workspace hostname.
    If not provided, the default value is determined by
      - the ``DATABRICKS_HOST`` environment variable if present, or
      - the hostname of the current Databricks workspace if running inside a Databricks notebook
        attached to an interactive 
    """

    api_token: str = Field(default_factory=get_default_api_token)
    """Databricks personal access token.
    If not provided, the default value is determined by
      - the ``DATABRICKS_API_TOKEN`` environment variable if present, or
      - an automatically generated temporary token if running inside a Databricks notebook.
    """

    endpoint_name: Optional[str] = None
    """Name of the model serving endpont.
    You must specify the endpoint name to connect to a model serving endpoint.
    """

    cluster_id: Optional[str] = None
    """ID of the cluster if connecting to 
    Cannot provide both `endpoint_name` and `cluster_id`.
    """

    cluster_driver_port: Optional[str|int] = None
    """The port number used by the HTTP server running on the cluster driver node.
    The server should listen on the driver IP address or simply ``0.0.0.0``.
    """

    transform_input_fn: Optional[Callable[..., Any]] = None
    """A function that transforms (prompt, stop) into a request dict that the endpoint accepts.
    For example, you can insert additional parameters like temperature.
    """

    transform_output_fn: Optional[Callable[..., str]] = None
    """A function that transfroms the output from the endpoint to generated text.
    """

    _client: DatabricksClientBase = PrivateAttr()

    class Config:
        extra = Extra.forbid
        underscore_attrs_are_private = True

    @validator("cluster_id", always=True)
    def set_cluster_id(cls, v, values):
        if values["endpoint_name"]:
            assert v is None, "Cannot provide both endpoint_name and cluster_id."
        elif not v:
            try:
                v = get_repl_context().clusterId
            except:
                pass
            assert v, "Neiter endpoint_name nor cluster_id was set. " \
                "And the cluster_id cannot be automatically determined."
        return v
    
    @validator("cluster_driver_port", always=True)
    def set_cluster_driver_port(cls, v, values):
        if values["endpoint_name"]:
            assert v is None, "Cannot set both endpoint_name and cluster_driver_port."
        else:
            assert v is not None, "Must set cluster_driver_port to connect to a cluster driver."
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.endpoint_name:
            self._client = DatabricksServingEndpointClient(
                host=self.host, api_token=self.api_token, endpoint_name=self.endpoint_name)
        elif self.cluster_id and self.cluster_driver_port:
            self._client = DatabricksClusterDriverProxyClient(
                host=self.host, api_token=self.api_token,
                cluster_id=self.cluster_id, cluster_driver_port=self.cluster_driver_port)
        else:
            raise ValueError("Must specify an endpoint name or cluster_id/driver_port.")

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "databricks_endpoint"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Queries the LLM endpoint with the given prompt and stop sequence."""

        request = {"prompt": prompt, "stop": stop}
        if self.transform_input_fn:
            request = self.transform_input_fn(**request)
        
        response = self._client.post(request)

        if self.transform_output_fn:
            response = self.transform_output_fn(response)

        return response
