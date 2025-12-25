import pytest

from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain_core.callbacks.manager import CallbackManager


def test_remove_handler() -> None:
    """Test removing handler does not raise an error on removal.

    An handler can be inheritable or not. This test checks that
    removing a handler does not raise an error if the handler
    is not inheritable.
    """
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()
    manager = BaseCallbackManager([handler1], inheritable_handlers=[handler2])
    manager.remove_handler(handler1)
    manager.remove_handler(handler2)


@pytest.mark.xfail(
    reason="TODO: #32028 merge() incorrectly mixes handlers and inheritable_handlers"
)
def test_merge_preserves_handler_distinction() -> None:
    """Test that merging managers preserves the distinction between handlers.

    This test verifies the correct behavior of the BaseCallbackManager.merge()
    method. When two managers are merged, their handlers and
    inheritable_handlers should be combined independently.

    Currently, it is expected to xfail until the issue is resolved.
    """
    h1 = BaseCallbackHandler()
    h2 = BaseCallbackHandler()
    ih1 = BaseCallbackHandler()
    ih2 = BaseCallbackHandler()

    m1 = BaseCallbackManager(handlers=[h1], inheritable_handlers=[ih1])
    m2 = BaseCallbackManager(handlers=[h2], inheritable_handlers=[ih2])

    merged = m1.merge(m2)

    assert set(merged.handlers) == {h1, h2}
    assert set(merged.inheritable_handlers) == {ih1, ih2}


def test_configure_tracing_keyword_only() -> None:
    """Test that tracing parameter must be passed as keyword-only argument.

    This test verifies that the tracing parameter in CallbackManager.configure()
    is keyword-only and cannot be passed as a positional argument.
    """
    # This should work - tracing as keyword argument
    manager = CallbackManager.configure(tracing=True)
    assert isinstance(manager, CallbackManager)

    # This should work - tracing with default value (False)
    manager = CallbackManager.configure()
    assert isinstance(manager, CallbackManager)

    # This should work - all parameters as keyword arguments
    manager = CallbackManager.configure(
        inheritable_callbacks=None,
        local_callbacks=None,
        verbose=False,
        inheritable_tags=["tag1"],
        local_tags=["tag2"],
        inheritable_metadata={"key": "value"},
        local_metadata={"local_key": "local_value"},
        tracing=False,
    )
    assert isinstance(manager, CallbackManager)

    # This should fail - tracing as positional argument
    # Using noqa/type:ignore to suppress linting since we're testing incorrect usage
    with pytest.raises(
        TypeError, match="takes from 1 to 8 positional arguments but 9 were given"
    ):
        CallbackManager.configure(  # type: ignore[misc]
            None,  # inheritable_callbacks
            None,  # local_callbacks
            False,  # verbose  # noqa: FBT003
            None,  # inheritable_tags
            None,  # local_tags
            None,  # inheritable_metadata
            None,  # local_metadata
            False,  # tracing - should fail as positional  # noqa: FBT003
        )


def test_configure_with_tracing_enabled() -> None:
    """Test that configure works correctly with tracing enabled.

    This test verifies that the tracing parameter is properly handled
    when set to True.
    """
    # Configure with tracing enabled
    manager = CallbackManager.configure(tracing=True)
    assert isinstance(manager, CallbackManager)

    # Verify that handlers are set up (when tracing is enabled, a tracer
    # should be added). Note: The actual tracer setup depends on environment
    # variables like LANGCHAIN_TRACING_V2. So we just verify the manager is
    # created successfully.
    assert manager is not None


def test_configure_with_callbacks_and_tracing() -> None:
    """Test configure with both callbacks and tracing parameter.

    This test verifies that the configure method works correctly when
    both callbacks and the tracing parameter are provided.
    """
    handler = BaseCallbackHandler()

    manager = CallbackManager.configure(
        inheritable_callbacks=[handler],
        tracing=False,
    )

    assert isinstance(manager, CallbackManager)
    assert handler in manager.handlers
