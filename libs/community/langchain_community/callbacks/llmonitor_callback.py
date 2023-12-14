from langchain_community.callbacks.lunary_callback import (
    LunaryCallbackHandler,
    identify,
)


class LLMonitorCallbackHandler(LunaryCallbackHandler):
    """LLMonitorCallbackHandler is deprecated, use LunaryCallbackHandler instead.
    ```
    from langchain_community.callbacks.lunary_callback import LunaryCallbackHandler
    ```
    """

    pass


__all__ = [
    "identify",
    "LLMonitorCallbackHandler",
]
