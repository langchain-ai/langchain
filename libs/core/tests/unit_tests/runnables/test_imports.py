from langchain_core.runnables import __all__

EXPECTED_ALL = [
    "chain",
    "AddableDict",
    "ConfigurableField",
    "ConfigurableFieldSingleOption",
    "ConfigurableFieldMultiOption",
    "ConfigurableFieldSpec",
    "ensure_config",
    "run_in_executor",
    "patch_config",
    "RouterInput",
    "RouterRunnable",
    "Runnable",
    "RunnableSerializable",
    "RunnableBinding",
    "RunnableBranch",
    "RunnableConfig",
    "RunnableGenerator",
    "RunnableLambda",
    "RunnableMap",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnableAssign",
    "RunnablePick",
    "RunnableSequence",
    "RunnableWithFallbacks",
    "RunnableWithMessageHistory",
    "StreamEventName",
    "get_config_list",
    "aadd",
    "add",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_imports_for_specific_funcs() -> None:
    """Test that a few specific imports in more internal namespaces."""
    # create_model implementation has been moved to langchain_core.utils.pydantic
    from langchain_core.runnables.utils import (  # type: ignore[attr-defined] # noqa: F401,PLC0415
        create_model,
    )
