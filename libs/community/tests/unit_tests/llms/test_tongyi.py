from langchain_community.llms import Tongyi


def test_tongyi_with_param_alias() -> None:
    """Test tongyi parameters alias"""
    llm = Tongyi(model="qwen-max", api_key="your-api_key")  # type: ignore[call-arg]
    assert llm.model_name == "qwen-max"
    assert llm.dashscope_api_key == "your-api_key"
