"""test Databricks LLM"""
import pickle
from typing import Any, Dict

from pytest import MonkeyPatch

from langchain_community.llms.databricks import Databricks


class MockDatabricksServingEndpointClient:
    def __init__(
        self,
        host: str,
        api_token: str,
        endpoint_name: str,
        databricks_uri: str,
        task: str,
    ):
        self.host = host
        self.api_token = api_token
        self.endpoint_name = endpoint_name
        self.databricks_uri = databricks_uri
        self.task = task


def transform_input(**request: Any) -> Dict[str, Any]:
    request["messages"] = [{"role": "user", "content": request["prompt"]}]
    del request["prompt"]
    return request


def test_serde_transform_input_fn(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        MockDatabricksServingEndpointClient,
    )
    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")

    llm = Databricks(
        endpoint_name="databricks-mixtral-8x7b-instruct",
        transform_input_fn=transform_input,
    )
    params = llm._default_params
    pickled_string = pickle.dumps(transform_input).hex()
    assert params["transform_input_fn"] == pickled_string
