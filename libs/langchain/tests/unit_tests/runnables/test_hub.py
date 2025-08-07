from typing import Any
from unittest.mock import Mock, patch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

from langchain.runnables.hub import HubRunnable


@patch("langchain.hub.pull")
def test_hub_runnable(mock_pull: Mock) -> None:
    mock_pull.return_value = ChatPromptTemplate.from_messages(
        [
            ("system", "a"),
            ("user", "b"),
        ],
    )

    basic: HubRunnable = HubRunnable("efriis/my-prompt")
    bound = basic.bound
    assert isinstance(bound, ChatPromptTemplate)
    assert len(bound.messages) == 2


repo_dict = {
    "efriis/my-prompt-1": ChatPromptTemplate.from_messages(
        [
            ("system", "a"),
            ("user", "1"),
        ],
    ),
    "efriis/my-prompt-2": ChatPromptTemplate.from_messages(
        [
            ("system", "a"),
            ("user", "2"),
        ],
    ),
}


def repo_lookup(owner_repo_commit: str, **_: Any) -> ChatPromptTemplate:
    return repo_dict[owner_repo_commit]


@patch("langchain.hub.pull")
def test_hub_runnable_configurable_alternative(mock_pull: Mock) -> None:
    mock_pull.side_effect = repo_lookup

    original: HubRunnable = HubRunnable("efriis/my-prompt-1")
    obj_a1 = original.configurable_alternatives(
        ConfigurableField(id="owner_repo_commit", name="Hub ID"),
        default_key="a1",
        a2=HubRunnable("efriis/my-prompt-2"),
    )

    obj_a2 = obj_a1.with_config(configurable={"owner_repo_commit": "a2"})

    templated = obj_a1.invoke({})
    message_a1 = templated.messages[1]
    assert message_a1.content == "1"

    templated_2 = obj_a2.invoke({})
    message_a2 = templated_2.messages[1]
    assert message_a2.content == "2"


@patch("langchain.hub.pull")
def test_hub_runnable_configurable_fields(mock_pull: Mock) -> None:
    mock_pull.side_effect = repo_lookup

    original: HubRunnable = HubRunnable("efriis/my-prompt-1")
    obj_configurable = original.configurable_fields(
        owner_repo_commit=ConfigurableField(id="owner_repo_commit", name="Hub ID"),
    )

    templated_1 = obj_configurable.invoke({})
    assert templated_1.messages[1].content == "1"

    templated_2 = obj_configurable.with_config(
        configurable={"owner_repo_commit": "efriis/my-prompt-2"},
    ).invoke({})
    assert templated_2.messages[1].content == "2"
