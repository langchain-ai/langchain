"""test Databricks LLM"""

from pathlib import Path
from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

from langchain_community.llms.databricks import (
    Databricks,
    _load_pickled_fn_from_hex_string,
)
from langchain_community.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


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
    fn = _load_pickled_fn_from_hex_string(
        data=params["transform_input_fn"],
        allow_dangerous_deserialization=True,
    )
    assert fn(**request) == transform_input(**request)


def test_saving_loading_llm(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        MockDatabricksServingEndpointClient,
    )
    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")

    llm = Databricks(
        endpoint_name="chat",
        temperature=0.1,
    )
    llm.save(file_path=tmp_path / "databricks.yaml")

    loaded_llm = load_llm(tmp_path / "databricks.yaml")
    assert_llm_equality(llm, loaded_llm)


@pytest.mark.requires("cloudpickle")
def test_saving_loading_llm_dangerous_serde_check(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        MockDatabricksServingEndpointClient,
    )
    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")

    llm1 = Databricks(
        endpoint_name="chat",
        temperature=0.1,
        transform_input_fn=lambda x, y, **kwargs: {},
    )
    llm1.save(file_path=tmp_path / "databricks1.yaml")

    with pytest.raises(ValueError, match="This code relies on the pickle module."):
        load_llm(tmp_path / "databricks1.yaml")

    load_llm(tmp_path / "databricks1.yaml", allow_dangerous_deserialization=True)

    llm2 = Databricks(
        endpoint_name="chat", temperature=0.1, transform_output_fn=lambda x: "test"
    )
    llm2.save(file_path=tmp_path / "databricks2.yaml")

    with pytest.raises(ValueError, match="This code relies on the pickle module."):
        load_llm(tmp_path / "databricks2.yaml")

    load_llm(tmp_path / "databricks2.yaml", allow_dangerous_deserialization=True)
