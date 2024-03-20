import pytest

from langchain.callbacks import manager
from langchain.callbacks.manager import __all__

EXPECTED_ALL = [
    "BaseRunManager",
    "RunManager",
    "ParentRunManager",
    "AsyncRunManager",
    "AsyncParentRunManager",
    "CallbackManagerForLLMRun",
    "AsyncCallbackManagerForLLMRun",
    "CallbackManagerForChainRun",
    "AsyncCallbackManagerForChainRun",
    "CallbackManagerForToolRun",
    "AsyncCallbackManagerForToolRun",
    "CallbackManagerForRetrieverRun",
    "AsyncCallbackManagerForRetrieverRun",
    "CallbackManager",
    "CallbackManagerForChainGroup",
    "AsyncCallbackManager",
    "AsyncCallbackManagerForChainGroup",
    "tracing_enabled",
    "tracing_v2_enabled",
    "collect_runs",
    "atrace_as_chain_group",
    "trace_as_chain_group",
    "handle_event",
    "ahandle_event",
    "env_var_is_set",
    "Callbacks",
]

EXPECTED_DEPRECATED_IMPORTS = [
    "get_openai_callback",
    "wandb_tracing_enabled",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(manager, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(manager, "foo")
