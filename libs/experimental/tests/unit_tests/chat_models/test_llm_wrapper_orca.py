import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_experimental.chat_models import Orca
from tests.unit_tests.chat_models.test_llm_wrapper_llama2chat import FakeLLM


@pytest.fixture
def model() -> Orca:
    return Orca(llm=FakeLLM())


@pytest.fixture
def model_cfg_sys_msg() -> Orca:
    return Orca(llm=FakeLLM(), system_message=SystemMessage(content="sys-msg"))


def test_prompt(model: Orca) -> None:
    messages = [
        SystemMessage(content="sys-msg"),
        HumanMessage(content="usr-msg-1"),
        AIMessage(content="ai-msg-1"),
        HumanMessage(content="usr-msg-2"),
    ]

    actual = model.invoke(messages).content  # type: ignore
    expected = "### System:\nsys-msg\n\n### User:\nusr-msg-1\n\n### Assistant:\nai-msg-1\n\n### User:\nusr-msg-2\n\n"  # noqa: E501

    assert actual == expected
