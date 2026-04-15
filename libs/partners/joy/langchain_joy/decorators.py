"""Decorators for Joy Trust verification."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from langchain_joy.callback import JoyTrustVerificationError
from langchain_joy.client import JoyTrustClient

F = TypeVar("F", bound=Callable[..., Any])


def require_trust(
    *,
    min_score: float = 1.5,
    agent_id_param: str = "agent_id",
    fail_open: bool = False,
    api_key: str | None = None,
) -> Callable[[F], F]:
    """Decorator to require minimum trust score for a function.

    Verifies the agent specified by agent_id_param meets the minimum
    trust threshold before executing the function.

    Example:
        >>> @require_trust(min_score=2.0, agent_id_param="target")
        ... def delegate_task(target: str, task: str) -> str:
        ...     return external_agent.run(task)
        ...
        >>> delegate_task("ag_trusted", "do something")  # Works
        >>> delegate_task("ag_untrusted", "do something")  # Raises error

    Args:
        min_score: Minimum trust score required.
        agent_id_param: Name of parameter containing agent ID.
        fail_open: If True, allow on errors. If False, block on errors.
        api_key: Optional API key for higher rate limits.

    Returns:
        Decorated function that verifies trust before execution.

    Raises:
        JoyTrustVerificationError: If trust verification fails.
    """
    client = JoyTrustClient(api_key=api_key)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract agent_id from kwargs
            agent_id = kwargs.get(agent_id_param)

            # If not in kwargs, try positional args using function signature
            if agent_id is None:
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if agent_id_param in params:
                    idx = params.index(agent_id_param)
                    if idx < len(args):
                        agent_id = args[idx]

            if agent_id is None:
                if fail_open:
                    return func(*args, **kwargs)
                raise JoyTrustVerificationError(
                    f"No agent ID found in parameter '{agent_id_param}'",
                )

            # Verify trust
            try:
                result = client.verify_trust(str(agent_id), min_trust=min_score)
                if not result["meets_threshold"]:
                    raise JoyTrustVerificationError(
                        f"Agent {agent_id} trust score {result['trust_score']} "
                        f"below threshold {min_score}",
                        agent_id=str(agent_id),
                        trust_score=result["trust_score"],
                        threshold=min_score,
                    )
            except JoyTrustVerificationError:
                raise
            except Exception as e:
                if fail_open:
                    pass  # Allow execution
                else:
                    raise JoyTrustVerificationError(
                        f"Trust verification failed: {e}",
                        agent_id=str(agent_id),
                    ) from e

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
