from typing import Any, Callable, Dict, List, Tuple


def register(key: str, _registry: Dict[str, Tuple[Any, List[str]]]) -> Any:
    """Add a class/function to a registry with required keyword arguments.

    ``_registry`` is a dictionary mapping from a key to a tuple of the class/function
    and a list of required keyword arguments, if keyword arguments are passed. Otherwise
    it is a dictionary mapping from a key to the class/function.
    """

    def _register_cls(cls: Any, required_kwargs: List = None) -> Any:
        if key in _registry:
            raise KeyError(f"{cls} already registered as {key}")
        _registry[key] = cls if required_kwargs is None else (cls, required_kwargs)
        return cls

    return _register_cls


AGENT_TO_CLASS: Dict[str, Any] = {}


def register_agent(key: str) -> Callable:
    """Register an agent."""

    def _register_agent_cls(cls: Any) -> Callable:
        register(key, AGENT_TO_CLASS)(cls=cls)
        return cls

    return _register_agent_cls


_TOOLS: Dict[str, Tuple[Callable, List[str]]] = {}
_LLM_TOOLS: Dict[str, Tuple[Callable, List[str]]] = {}


def register_tool(key: str, required_kwargs: List[str] = []) -> Callable:
    """Register a tool."""

    def _register_tool_cls(cls: Any) -> Callable:
        register(key, _TOOLS)(cls=cls, required_kwargs=required_kwargs)
        return cls

    return _register_tool_cls


def register_llm_tool(key: str, required_kwargs: List[str] = []) -> Callable:
    """Register an LLM tool."""

    def _register_llm_tool_cls(cls: Any) -> Callable:
        register(key, _LLM_TOOLS)(cls=cls, required_kwargs=required_kwargs)
        return cls

    return _register_llm_tool_cls
