import pytest
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_experimental.chat_models import Mixtral
from tests.unit_tests.chat_models.test_llm_wrapper_llama2chat import FakeLLM


@pytest.fixture
def model() -> Mixtral:
    return Mixtral(llm=FakeLLM())


@pytest.fixture
def model_cfg_sys_msg() -> Mixtral:
    return Mixtral(llm=FakeLLM(), system_message=SystemMessage(content="sys-msg"))


def test_prompt(model: Mixtral) -> None:
    messages = [
        SystemMessage(content="sys-msg"),
        HumanMessage(content="usr-msg-1"),
        AIMessage(content="ai-msg-1"),
        HumanMessage(content="usr-msg-2"),
    ]

    actual = model.invoke(messages).content  # type: ignore
    expected = (
        "<s>[INST] sys-msg\nusr-msg-1 [/INST] ai-msg-1 </s> [INST] usr-msg-2 [/INST]"
    )

    assert actual == expected
