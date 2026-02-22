from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager


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
