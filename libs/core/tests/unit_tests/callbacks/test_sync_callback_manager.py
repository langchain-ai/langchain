from langchain_core.callbacks.manager import BaseCallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler


def test_remove_handler():
    """Test adding and removing a handler."""
    handler = BaseCallbackHandler()
    manager = BaseCallbackManager(handlers=[handler])
    manager.remove_handler(handler)
