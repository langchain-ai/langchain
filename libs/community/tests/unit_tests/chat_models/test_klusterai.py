from langchain_community.chat_models import ChatKlusterAi


def test_klusterai_model_name_param() -> None:
    llm = ChatKlusterAi(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_klusterai_model_param() -> None:
    llm = ChatKlusterAi(model="foo")
    assert llm.model_name == "foo"
