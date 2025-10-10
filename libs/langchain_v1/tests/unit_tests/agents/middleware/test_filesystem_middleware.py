from langchain.agents.middleware.filesystem import (
    FileData,
    FilesystemState,
    FilesystemMiddleware,
    FILESYSTEM_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT,
)
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


class TestFilesystem:
    def test_init_local(self):
        middleware = FilesystemMiddleware(use_longterm_memory=False)
        assert middleware.use_longterm_memory is False
        assert middleware.system_prompt_extension == FILESYSTEM_SYSTEM_PROMPT
        assert len(middleware.tools) == 4

    def test_init_longterm(self):
        middleware = FilesystemMiddleware(use_longterm_memory=True)
        assert middleware.use_longterm_memory is True
        assert middleware.system_prompt_extension == (
            FILESYSTEM_SYSTEM_PROMPT + FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT
        )
        assert len(middleware.tools) == 4

    def test_init_custom_system_prompt_shortterm(self):
        middleware = FilesystemMiddleware(
            use_longterm_memory=False, system_prompt_extension="Custom system prompt"
        )
        assert middleware.use_longterm_memory is False
        assert middleware.system_prompt_extension == "Custom system prompt"
        assert len(middleware.tools) == 4

    def test_init_custom_system_prompt_longterm(self):
        middleware = FilesystemMiddleware(
            use_longterm_memory=True, system_prompt_extension="Custom system prompt"
        )
        assert middleware.use_longterm_memory is True
        assert middleware.system_prompt_extension == "Custom system prompt"
        assert len(middleware.tools) == 4

    def test_init_custom_tool_descriptions_shortterm(self):
        middleware = FilesystemMiddleware(
            use_longterm_memory=False, custom_tool_descriptions={"ls": "Custom ls tool description"}
        )
        assert middleware.use_longterm_memory is False
        assert middleware.system_prompt_extension == FILESYSTEM_SYSTEM_PROMPT
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_init_custom_tool_descriptions_longterm(self):
        middleware = FilesystemMiddleware(
            use_longterm_memory=True, custom_tool_descriptions={"ls": "Custom ls tool description"}
        )
        assert middleware.use_longterm_memory is True
        assert middleware.system_prompt_extension == (
            FILESYSTEM_SYSTEM_PROMPT + FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT
        )
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_ls_shortterm(self):
        state = FilesystemState(
            messages=[],
            files={
                "test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware(use_longterm_memory=False)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke({"state": state})
        assert result == ["test.txt", "test2.txt"]

    def test_ls_shortterm_with_path(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/charmander.txt": FileData(
                    content=["Ember"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/water/squirtle.txt": FileData(
                    content=["Water"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware(use_longterm_memory=False)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke({"state": state, "path": "pokemon/"})
        assert "/pokemon/test2.txt" in result
        assert "/pokemon/charmander.txt" in result
