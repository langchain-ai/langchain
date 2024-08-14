"""Test SparkLLM."""

from langchain_core.outputs import LLMResult

from langchain_community.llms.sparkllm import SparkLLM


def test_call() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_spark_llm_with_param_alias() -> None:
    """Test SparkLLM with parameters alias."""
    llm = SparkLLM(  # type: ignore[call-arg]
        app_id="your-app-id",
        api_key="your-api-key",
        api_secret="your-api-secret",
        model="Spark4.0 Ultra",
        api_url="your-api-url",
        timeout=20,
    )
    assert llm.spark_app_id == "your-app-id"
    assert llm.spark_api_key == "your-api-key"
    assert llm.spark_api_secret == "your-api-secret"
    assert llm.spark_llm_domain == "Spark4.0 Ultra"
    assert llm.spark_api_url == "your-api-url"
    assert llm.request_timeout == 20


def test_spark_llm_with_stream() -> None:
    """Test SparkLLM with stream."""
    llm = SparkLLM()  # type: ignore[call-arg]
    for chunk in llm.stream("你好呀"):
        assert isinstance(chunk, str)
