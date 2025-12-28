"""Tests for existing middleware classes using depends_on parameter."""

from langchain.agents.middleware.context_editing import ContextEditingMiddleware
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain.agents.middleware.pii import PIIMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.middleware.tool_emulator import LLMToolEmulator
from langchain.agents.middleware.tool_retry import ToolRetryMiddleware
from langchain.agents.middleware.tool_selection import LLMToolSelectorMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_model_retry_with_depends_on() -> None:
    """Test that ModelRetryMiddleware accepts depends_on parameter."""
    middleware = ModelRetryMiddleware(max_retries=3, depends_on=(PIIMiddleware("email"),))
    assert len(middleware.depends_on) == 1
    assert isinstance(middleware.depends_on[0], PIIMiddleware)


def test_tool_retry_with_depends_on() -> None:
    """Test that ToolRetryMiddleware accepts depends_on parameter."""
    middleware = ToolRetryMiddleware(max_retries=3, depends_on=(ModelRetryMiddleware,))
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_pii_with_depends_on() -> None:
    """Test that PIIMiddleware accepts depends_on parameter."""
    middleware = PIIMiddleware("email", depends_on=(ModelRetryMiddleware,))
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_model_fallback_with_depends_on() -> None:
    """Test that ModelFallbackMiddleware accepts depends_on parameter."""
    model1 = FakeToolCallingModel()
    model2 = FakeToolCallingModel()
    middleware = ModelFallbackMiddleware(
        model1,
        model2,
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_tool_selection_with_depends_on() -> None:
    """Test that LLMToolSelectorMiddleware accepts depends_on parameter."""
    middleware = LLMToolSelectorMiddleware(
        max_tools=5,
        depends_on=(PIIMiddleware("email"),),
    )
    assert len(middleware.depends_on) == 1
    assert isinstance(middleware.depends_on[0], PIIMiddleware)


def test_todo_with_depends_on() -> None:
    """Test that TodoListMiddleware accepts depends_on parameter."""
    middleware = TodoListMiddleware(depends_on=(ModelRetryMiddleware,))
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_summarization_with_depends_on() -> None:
    """Test that SummarizationMiddleware accepts depends_on parameter."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model,
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_context_editing_with_depends_on() -> None:
    """Test that ContextEditingMiddleware accepts depends_on parameter."""
    middleware = ContextEditingMiddleware(depends_on=(ModelRetryMiddleware,))
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_file_search_with_depends_on() -> None:
    """Test that FilesystemFileSearchMiddleware accepts depends_on parameter."""
    middleware = FilesystemFileSearchMiddleware(
        root_path="/test",
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_tool_emulator_with_depends_on() -> None:
    """Test that LLMToolEmulator accepts depends_on parameter."""
    middleware = LLMToolEmulator(depends_on=(ModelRetryMiddleware,))
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_mixed_depends_on_types() -> None:
    """Test that middleware can accept both classes and instances in depends_on."""
    pii_instance = PIIMiddleware("email")
    middleware = ModelRetryMiddleware(
        max_retries=3,
        depends_on=(pii_instance, ToolRetryMiddleware),
    )
    assert len(middleware.depends_on) == 2
    assert middleware.depends_on[0] is pii_instance
    assert middleware.depends_on[1] == ToolRetryMiddleware


def test_human_in_the_loop_with_depends_on() -> None:
    """Test that HumanInTheLoopMiddleware accepts depends_on parameter."""
    from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"search": True},
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_model_call_limit_with_depends_on() -> None:
    """Test that ModelCallLimitMiddleware accepts depends_on parameter."""
    from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware

    middleware = ModelCallLimitMiddleware(
        thread_limit=10,
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_tool_call_limit_with_depends_on() -> None:
    """Test that ToolCallLimitMiddleware accepts depends_on parameter."""
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

    middleware = ToolCallLimitMiddleware(
        thread_limit=5,
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware


def test_shell_tool_with_depends_on() -> None:
    """Test that ShellToolMiddleware accepts depends_on parameter."""
    from langchain.agents.middleware.shell_tool import ShellToolMiddleware

    middleware = ShellToolMiddleware(
        workspace_root="/test",
        depends_on=(ModelRetryMiddleware,),
    )
    assert len(middleware.depends_on) == 1
    assert middleware.depends_on[0] == ModelRetryMiddleware
