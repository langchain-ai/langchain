import os

import pytest
from langchain_core.pydantic_v1 import ValidationError

from langchain_community.embeddings import SparkLLMTextEmbeddings


def test_sparkllm_initialization_by_alias() -> None:
    # Effective initialization
    embeddings = SparkLLMTextEmbeddings(
        app_id="your-app-id", api_key="your-api-key", api_secret="your-api-secret"
    )
    assert embeddings.spark_app_id.get_secret_value() == "your-app-id"
    assert embeddings.spark_api_key.get_secret_value() == "your-api-key"
    assert embeddings.spark_api_secret.get_secret_value() == "your-api-secret"


def test_initialization_parameters_from_env() -> None:
    # Setting environment variable
    os.environ["SPARK_APP_ID"] = "your-app-id"
    os.environ["SPARK_API_KEY"] = "your-api-key"
    os.environ["SPARK_API_SECRET"] = "your-api-secret"

    # Effective initialization
    embeddings = SparkLLMTextEmbeddings()
    assert embeddings.spark_app_id.get_secret_value() == "your-app-id"
    assert embeddings.spark_api_key.get_secret_value() == "your-api-key"
    assert embeddings.spark_api_secret.get_secret_value() == "your-api-secret"

    # Environment variable missing
    del os.environ["SPARK_APP_ID"]
    with pytest.raises(ValidationError):
        SparkLLMTextEmbeddings()
