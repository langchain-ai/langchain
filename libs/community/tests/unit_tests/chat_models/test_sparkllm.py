from langchain_community.chat_models.sparkllm import ChatSparkLLM


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatSparkLLM(
            spark_app_id="123",
            spark_api_secret="test",
            api_key="secret",
            temperature=0.5,
            timeout=30,
        ),
        ChatSparkLLM(
            spark_app_id="123",
            spark_api_secret="test",
            spark_api_key="secret",
            request_timeout=30,
        ),
    ]:
        assert model.request_timeout == 30
        assert model.spark_api_key == "secret"
        assert model.temperature == 0.5
