"""test Databricks LLM"""
from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

from langchain_community.llms.databricks import (
    Databricks,
    _load_pickled_fn_from_hex_string,
)


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


@pytest.mark.requires("cloudpickle")
def test_serde_transform_input_fn(monkeypatch: MonkeyPatch) -> None:
    import cloudpickle

    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        MockDatabricksServingEndpointClient,
    )
    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")

    llm = Databricks(
        endpoint_name="some_end_point_name",  # Value should not matter for this test
        transform_input_fn=transform_input,
        allow_dangerous_deserialization=True,
    )
    params = llm._default_params
    pickled_string = cloudpickle.dumps(transform_input).hex()
    assert params["transform_input_fn"] == pickled_string

    request = {"prompt": "What is the meaning of life?"}
    fn = _load_pickled_fn_from_hex_string(params["transform_input_fn"])
    assert fn(**request) == transform_input(**request)
