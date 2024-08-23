from typing import Any, List, Optional

import pytest
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_experimental.chat_models import Llama2Chat
from langchain_experimental.chat_models.llm_wrapper import DEFAULT_SYSTEM_PROMPT


class FakeLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return prompt

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return prompt

    @property
    def _llm_type(self) -> str:
        return "fake-llm"


@pytest.fixture
def model() -> Llama2Chat:
    return Llama2Chat(llm=FakeLLM())


@pytest.fixture
def model_cfg_sys_msg() -> Llama2Chat:
    return Llama2Chat(llm=FakeLLM(), system_message=SystemMessage(content="sys-msg"))


def test_default_system_message(model: Llama2Chat) -> None:
    messages = [HumanMessage(content="usr-msg-1")]

    actual = model.invoke(messages).content  # type: ignore
    expected = (
        f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\nusr-msg-1 [/INST]"
    )

    assert actual == expected


def test_configured_system_message(
    model_cfg_sys_msg: Llama2Chat,
) -> None:
    messages = [HumanMessage(content="usr-msg-1")]

    actual = model_cfg_sys_msg.invoke(messages).content  # type: ignore
    expected = "<s>[INST] <<SYS>>\nsys-msg\n<</SYS>>\n\nusr-msg-1 [/INST]"

    assert actual == expected


async def test_configured_system_message_async(
    model_cfg_sys_msg: Llama2Chat,
) -> None:
    messages = [HumanMessage(content="usr-msg-1")]

    actual = await model_cfg_sys_msg.ainvoke(messages)  # type: ignore
    expected = "<s>[INST] <<SYS>>\nsys-msg\n<</SYS>>\n\nusr-msg-1 [/INST]"

    assert actual.content == expected


def test_provided_system_message(
    model_cfg_sys_msg: Llama2Chat,
) -> None:
    messages = [
        SystemMessage(content="custom-sys-msg"),
        HumanMessage(content="usr-msg-1"),
    ]

    actual = model_cfg_sys_msg.invoke(messages).content
    expected = "<s>[INST] <<SYS>>\ncustom-sys-msg\n<</SYS>>\n\nusr-msg-1 [/INST]"

    assert actual == expected


def test_human_ai_dialogue(model_cfg_sys_msg: Llama2Chat) -> None:
    messages = [
        HumanMessage(content="usr-msg-1"),
        AIMessage(content="ai-msg-1"),
        HumanMessage(content="usr-msg-2"),
        AIMessage(content="ai-msg-2"),
        HumanMessage(content="usr-msg-3"),
    ]

    actual = model_cfg_sys_msg.invoke(messages).content
    expected = (
        "<s>[INST] <<SYS>>\nsys-msg\n<</SYS>>\n\nusr-msg-1 [/INST] ai-msg-1 </s>"
        "<s>[INST] usr-msg-2 [/INST] ai-msg-2 </s><s>[INST] usr-msg-3 [/INST]"
    )

    assert actual == expected


def test_no_message(model: Llama2Chat) -> None:
    with pytest.raises(ValueError) as info:
        model.invoke([])

    assert info.value.args[0] == "at least one HumanMessage must be provided"


def test_ai_message_first(model: Llama2Chat) -> None:
    with pytest.raises(ValueError) as info:
        model.invoke([AIMessage(content="ai-msg-1")])

    assert (
        info.value.args[0]
        == "messages list must start with a SystemMessage or UserMessage"
    )


def test_human_ai_messages_not_alternating(model: Llama2Chat) -> None:
    messages = [
        HumanMessage(content="usr-msg-1"),
        HumanMessage(content="usr-msg-2"),
        HumanMessage(content="ai-msg-1"),
    ]

    with pytest.raises(ValueError) as info:
        model.invoke(messages)  # type: ignore

    assert info.value.args[0] == (
        "messages must be alternating human- and ai-messages, "
        "optionally prepended by a system message"
    )


def test_last_message_not_human_message(model: Llama2Chat) -> None:
    messages = [
        HumanMessage(content="usr-msg-1"),
        AIMessage(content="ai-msg-1"),
    ]

    with pytest.raises(ValueError) as info:
        model.invoke(messages)

    assert info.value.args[0] == "last message must be a HumanMessage"
