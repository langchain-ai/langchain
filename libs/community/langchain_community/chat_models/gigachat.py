def __getattr__(name: str = "") -> None:
    """Raise an error on import since is deprecated."""
    raise AttributeError("This module has been moved to langchain_gigachat module")


class GigaChat:
    def __init___(self) -> None:
        """Raise an error on import since is deprecated."""
        raise AttributeError("This module has been moved to langchain_gigachat module")
