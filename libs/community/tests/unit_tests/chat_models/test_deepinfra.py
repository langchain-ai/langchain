from langchain_community.chat_models import ChatDeepInfra


def test_deepinfra_model_name_param() -> None:
    llm = ChatDeepInfra(model_name="foo")
    assert llm.model_name == "foo"


def test_deepinfra_model_param() -> None:
    llm = ChatDeepInfra(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
