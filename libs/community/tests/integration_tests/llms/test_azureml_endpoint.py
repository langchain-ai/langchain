"""Test AzureML Endpoint wrapper."""

import json
import os
from pathlib import Path
from typing import Dict
from urllib.request import HTTPError

import pytest
from pydantic import ValidationError

from langchain_community.llms.azureml_endpoint import (
    AzureMLOnlineEndpoint,
    ContentFormatterBase,
    DollyContentFormatter,
    HFContentFormatter,
    OSSContentFormatter,
)
from langchain_community.llms.loading import load_llm


def test_gpt2_call() -> None:
    """Test valid call to GPT2."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=OSSContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)


def test_hf_call() -> None:
    """Test valid call to HuggingFace Foundation Model."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("HF_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("HF_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("HF_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=HFContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)


def test_dolly_call() -> None:
    """Test valid call to dolly-v2."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("DOLLY_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("DOLLY_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("DOLLY_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=DollyContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)


def test_custom_formatter() -> None:
    """Test ability to create a custom content formatter."""

    class CustomFormatter(ContentFormatterBase):
        content_type: str = "application/json"
        accepts: str = "application/json"

        def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:  # type: ignore[override]
            input_str = json.dumps(
                {
                    "inputs": [prompt],
                    "parameters": model_kwargs,
                    "options": {"use_cache": False, "wait_for_model": True},
                }
            )
            return input_str.encode("utf-8")

        def format_response_payload(self, output: bytes) -> str:  # type: ignore[override]
            response_json = json.loads(output)
            return response_json[0]["summary_text"]

    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("BART_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("BART_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("BART_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=CustomFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)


def test_missing_content_formatter() -> None:
    """Test AzureML LLM without a content_formatter attribute"""
    with pytest.raises(AttributeError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        )
        llm.invoke("Foo")


def test_invalid_request_format() -> None:
    """Test invalid request format."""

    class CustomContentFormatter(ContentFormatterBase):
        content_type: str = "application/json"
        accepts: str = "application/json"

        def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:  # type: ignore[override]
            input_str = json.dumps(
                {
                    "incorrect_input": {"input_string": [prompt]},
                    "parameters": model_kwargs,
                }
            )
            return str.encode(input_str)

        def format_response_payload(self, output: bytes) -> str:  # type: ignore[override]
            response_json = json.loads(output)
            return response_json[0]["0"]

    with pytest.raises(HTTPError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=CustomContentFormatter(),
        )
        llm.invoke("Foo")


def test_incorrect_url() -> None:
    """Testing AzureML Endpoint for an incorrect URL"""
    with pytest.raises(ValidationError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url="https://endpoint.inference.com",
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")


def test_incorrect_api_type() -> None:
    with pytest.raises(ValidationError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            endpoint_api_type="serverless",  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")


def test_incorrect_key() -> None:
    """Testing AzureML Endpoint for incorrect key"""
    with pytest.raises(HTTPError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key="incorrect-key",  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an AzureML Foundation Model LLM."""

    save_llm = AzureMLOnlineEndpoint(
        deployment_name="databricks-dolly-v2-12b-4",
        model_kwargs={"temperature": 0.03, "top_p": 0.4, "max_tokens": 200},
    )
    save_llm.save(file_path=tmp_path / "azureml.yaml")
    loaded_llm = load_llm(tmp_path / "azureml.yaml")

    assert loaded_llm == save_llm
