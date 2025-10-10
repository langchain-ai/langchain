from langchain.agents.middleware.filesystem import (
    FilesystemMiddleware,
    WRITE_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT,
)
from langchain.agents import create_agent
from langchain.agents.deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
import pytest
import uuid


class TestFilesystem:
    def test_create_deepagent_without_store_and_with_longterm_memory_should_fail(self):
        with pytest.raises(ValueError):
            deepagent = create_deep_agent(tools=[], use_longterm_memory=True)
            deepagent.invoke(
                {"messages": [HumanMessage(content="List all of the files in your filesystem?")]}
            )

    def test_filesystem_system_prompt_override(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            middleware=[
                FilesystemMiddleware(
                    use_longterm_memory=False,
                    system_prompt_extension="In every single response, you must say the word 'pokemon'! You love it!",
                )
            ],
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pokemon" in response["messages"][1].text.lower()

    def test_filesystem_system_prompt_override_with_longterm_memory(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            middleware=[
                FilesystemMiddleware(
                    use_longterm_memory=True,
                    system_prompt_extension="In every single response, you must say the word 'pokemon'! You love it!",
                )
            ],
            store=InMemoryStore(),
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pokemon" in response["messages"][1].text.lower()

    def test_filesystem_tool_prompt_override(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            middleware=[
                FilesystemMiddleware(
                    use_longterm_memory=False,
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert tools["write_file"].description == WRITE_FILE_TOOL_DESCRIPTION
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"

    def test_filesystem_tool_prompt_override_with_longterm_memory(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            middleware=[
                FilesystemMiddleware(
                    use_longterm_memory=True,
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
            store=InMemoryStore(),
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert (
            tools["write_file"].description
            == WRITE_FILE_TOOL_DESCRIPTION + WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT
        )
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"

    def test_longterm_memory_tools(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            middleware=[
                FilesystemMiddleware(
                    use_longterm_memory=True,
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        assert_longterm_mem_tools(agent, store)

    def test_longterm_memory_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
        assert_longterm_mem_tools(agent, store)

    def test_shortterm_memory_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_deep_agent(use_longterm_memory=False, checkpointer=checkpointer, store=store)
        assert_shortterm_mem_tools(agent)


def assert_longterm_mem_tools(agent, store):
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a haiku about Charmander to longterm memory in charmander.txt, use the word 'fiery'"
                )
            ]
        },
        config=config,
    )

    namespaces = store.list_namespaces()
    assert len(namespaces) == 1
    assert namespaces[0] == ("filesystem",)
    file_item = store.get(("filesystem",), "charmander.txt")
    assert file_item is not None
    assert file_item.key == "charmander.txt"

    config2 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Read the haiku about Charmander from longterm memory at charmander.txt"
                )
            ]
        },
        config=config2,
    )

    messages = response["messages"]
    read_file_message = next(
        message for message in messages if message.type == "tool" and message.name == "read_file"
    )
    assert "fiery" in read_file_message.content or "Fiery" in read_file_message.content

    config3 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in longterm memory")]},
        config=config3,
    )
    messages = response["messages"]
    ls_message = next(
        message for message in messages if message.type == "tool" and message.name == "ls"
    )
    assert "memories/charmander.txt" in ls_message.content

    config4 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Edit the haiku about Charmander in longterm memory to use the word 'ember'"
                )
            ]
        },
        config=config4,
    )
    file_item = store.get(("filesystem",), "charmander.txt")
    assert file_item is not None
    assert file_item.key == "charmander.txt"
    assert "ember" in file_item.value["content"] or "Ember" in file_item.value["content"]

    config5 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Read the haiku about Charmander from longterm memory at charmander.txt"
                )
            ]
        },
        config=config5,
    )
    messages = response["messages"]
    read_file_message = next(
        message for message in messages if message.type == "tool" and message.name == "read_file"
    )
    assert "ember" in read_file_message.content or "Ember" in read_file_message.content


def assert_shortterm_mem_tools(agent):
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a haiku about Charmander to charmander.txt, use the word 'fiery'"
                )
            ]
        },
        config=config,
    )
    files = response["files"]
    assert "charmander.txt" in files

    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander from charmander.txt")]},
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(
        message
        for message in reversed(messages)
        if message.type == "tool" and message.name == "read_file"
    )
    assert "fiery" in read_file_message.content or "Fiery" in read_file_message.content

    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in memory")]}, config=config
    )
    messages = response["messages"]
    ls_message = next(
        message for message in messages if message.type == "tool" and message.name == "ls"
    )
    assert "charmander.txt" in ls_message.content

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Edit the haiku about Charmander to use the word 'ember'")
            ]
        },
        config=config,
    )
    files = response["files"]
    assert "charmander.txt" in files
    assert "ember" in "\n".join(files["charmander.txt"]["content"]) or "Ember" in "\n".join(
        files["charmander.txt"]["content"]
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander at charmander.txt")]},
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(
        message
        for message in reversed(messages)
        if message.type == "tool" and message.name == "read_file"
    )
    assert "ember" in read_file_message.content or "Ember" in read_file_message.content
