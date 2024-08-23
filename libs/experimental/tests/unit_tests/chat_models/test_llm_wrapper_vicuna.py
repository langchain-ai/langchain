import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_experimental.chat_models import Vicuna
from tests.unit_tests.chat_models.test_llm_wrapper_llama2chat import FakeLLM


@pytest.fixture
def model() -> Vicuna:
    return Vicuna(llm=FakeLLM())


@pytest.fixture
def model_cfg_sys_msg() -> Vicuna:
    return Vicuna(llm=FakeLLM(), system_message=SystemMessage(content="sys-msg"))


def test_prompt(model: Vicuna) -> None:
    messages = [
        SystemMessage(content="sys-msg"),
        HumanMessage(content="usr-msg-1"),
        AIMessage(content="ai-msg-1"),
        HumanMessage(content="usr-msg-2"),
    ]

    actual = model.invoke(messages).content  # type: ignore
    expected = "sys-msg USER: usr-msg-1 ASSISTANT: ai-msg-1 </s>USER: usr-msg-2 "

    assert actual == expected
