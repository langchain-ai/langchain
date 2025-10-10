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
from langchain.agents._internal.file_utils import FileData
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

    def test_ls_longterm_without_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="List all of your files")],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(
            message for message in messages if message.type == "tool" and message.name == "ls"
        )
        assert "/pizza.txt" in ls_message.text
        assert "/pokemon/squirtle.txt" in ls_message.text
        assert "/memories/test.txt" in ls_message.text
        assert "/memories/pokemon/charmander.txt" in ls_message.text

    def test_ls_longterm_with_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="List all of your files in the /pokemon directory")
                ],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(
            message for message in messages if message.type == "tool" and message.name == "ls"
        )
        assert "/pokemon/squirtle.txt" in ls_message.text
        assert "/memories/pokemon/charmander.txt" not in ls_message.text

    def test_read_file_longterm_local_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from local memory")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "read_file"
        )
        assert read_file_message is not None
        assert "Goodbye world" in read_file_message.content

    def test_read_file_longterm_store_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from longterm memory")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "read_file"
        )
        assert read_file_message is not None
        assert "Hello world" in read_file_message.content

    def test_read_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read the contents of the file about charmander from longterm memory."
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        ai_msg_w_toolcall = next(
            message
            for message in messages
            if message.type == "ai"
            and any(
                tc["name"] == "read_file"
                and tc["args"]["file_path"] == "/memories/pokemon/charmander.txt"
                for tc in message.tool_calls
            )
        )
        assert ai_msg_w_toolcall is not None

    def test_write_file_longterm(self):
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Write a haiku about Charmander to longterm memory in /charmander.txt, use the word 'fiery'"
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "write_file"
        )
        assert write_file_message is not None
        file_item = store.get(("filesystem",), "/charmander.txt")
        assert file_item is not None
        assert any("fiery" in c for c in file_item.value["content"]) or any(
            "Fiery" in c for c in file_item.value["content"]
        )

    def test_write_file_fail_already_exists_in_store(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/charmander.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Write a haiku about Charmander to longterm memory in /charmander.txt, use the word 'fiery'"
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "write_file"
        )
        assert write_file_message is not None
        assert "Cannot write" in write_file_message.content

    def test_write_file_fail_already_exists_in_local(self):
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Write a haiku about Charmander to /charmander.txt, use the word 'fiery'"
                    )
                ],
                "files": {
                    "/charmander.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "write_file"
        )
        assert write_file_message is not None
        assert "Cannot write" in write_file_message.content

    def test_edit_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/charmander.txt",
            {
                "content": ["The fire burns brightly. The fire burns hot."],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
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
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Edit the longterm memory file about charmander, to replace all instances of the word 'fire' with 'embers'"
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        edit_file_message = next(
            message
            for message in messages
            if message.type == "tool" and message.name == "edit_file"
        )
        assert edit_file_message is not None
        assert store.get(("filesystem",), "/charmander.txt").value["content"] == [
            "The embers burns brightly. The embers burns hot."
        ]

    def test_longterm_memory_multiple_tools(self):
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

    def test_longterm_memory_multiple_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
        assert_longterm_mem_tools(agent, store)

    def test_shortterm_memory_multiple_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_deep_agent(use_longterm_memory=False, checkpointer=checkpointer, store=store)
        assert_shortterm_mem_tools(agent)


# Take actions on multiple threads to test longterm memory
def assert_longterm_mem_tools(agent, store):
    # Write a longterm memory file
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a haiku about Charmander to longterm memory in /charmander.txt, use the word 'fiery'"
                )
            ]
        },
        config=config,
    )
    namespaces = store.list_namespaces()
    assert len(namespaces) == 1
    assert namespaces[0] == ("filesystem",)
    file_item = store.get(("filesystem",), "/charmander.txt")
    assert file_item is not None
    assert file_item.key == "/charmander.txt"

    # Read the longterm memory file
    config2 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Read the haiku about Charmander from longterm memory at /charmander.txt"
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

    # List all of the files in longterm memory
    config3 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in longterm memory")]},
        config=config3,
    )
    messages = response["messages"]
    ls_message = next(
        message for message in messages if message.type == "tool" and message.name == "ls"
    )
    assert "/memories/charmander.txt" in ls_message.content

    # Edit the longterm memory file
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
    file_item = store.get(("filesystem",), "/charmander.txt")
    assert file_item is not None
    assert file_item.key == "/charmander.txt"
    assert any("ember" in c for c in file_item.value["content"]) or any(
        "Ember" in c for c in file_item.value["content"]
    )

    # Read the longterm memory file
    config5 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Read the haiku about Charmander from longterm memory at /charmander.txt"
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
    # Write a shortterm memory file
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a haiku about Charmander to /charmander.txt, use the word 'fiery'"
                )
            ]
        },
        config=config,
    )
    files = response["files"]
    assert "/charmander.txt" in files

    # Read the shortterm memory file
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Read the haiku about Charmander from /charmander.txt")
            ]
        },
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(
        message
        for message in reversed(messages)
        if message.type == "tool" and message.name == "read_file"
    )
    assert "fiery" in read_file_message.content or "Fiery" in read_file_message.content

    # List all of the files in shortterm memory
    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in your filesystem")]},
        config=config,
    )
    messages = response["messages"]
    ls_message = next(
        message for message in messages if message.type == "tool" and message.name == "ls"
    )
    assert "/charmander.txt" in ls_message.content

    # Edit the shortterm memory file
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Edit the haiku about Charmander to use the word 'ember'")
            ]
        },
        config=config,
    )
    files = response["files"]
    assert "/charmander.txt" in files
    assert any("ember" in c for c in files["/charmander.txt"]["content"]) or any(
        "Ember" in c for c in files["/charmander.txt"]["content"]
    )

    # Read the shortterm memory file
    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander at /charmander.txt")]},
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(
        message
        for message in reversed(messages)
        if message.type == "tool" and message.name == "read_file"
    )
    assert "ember" in read_file_message.content or "Ember" in read_file_message.content
