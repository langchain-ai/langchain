"""Base Guard class."""
from typing import Any, Callable, Tuple, Union


class BaseGuard:
    """The Guard class is a decorator that can be applied to any chain or agent.

    Can be used to either throw an error or recursively call the chain or agent
    when the output of said chain or agent violates the rules of the guard.
    The BaseGuard alone does nothing but can be subclassed and the resolve_guard
    function overwritten to create more specific guards.

    Args:
        retries (int, optional): The number of times the chain or agent should be
            called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum number
            of retries has been exceeded.
    """

    def __init__(self, retries: int = 0, *args: Any, **kwargs: Any) -> None:
        """Initialize with number of retries."""
        self.retries = retries

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Determine if guard was violated (if response should be blocked).

        Can be overwritten when subclassing to expand on guard functionality

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated
                    (if guard was violated).
        """
        return False, ""

    def handle_violation(self, message: str, *args: Any, **kwargs: Any) -> Exception:
        """Handle violation of guard.

        Args:
            message (str): the message to be displayed when the guard is violated.

        Raises:
            Exception: the message passed to the function.
        """
        raise Exception(message)

    def __call__(self, func: Callable) -> Callable:
        """Create wrapper to be returned."""

        def wrapper(*args: Any, **kwargs: Any) -> Union[str, Exception]:
            """Create wrapper to return."""
            if self.retries < 0:
                raise Exception("Restriction violated. Maximum retries exceeded.")
            try:
                llm_response = func(*args, **kwargs)
                guard_result, violation_message = self.resolve_guard(llm_response)
                if guard_result:
                    return self.handle_violation(violation_message)
                else:
                    return llm_response
            except Exception as e:
                self.retries = self.retries - 1
                # Check retries to avoid infinite recursion if exception is something
                # other than a violation of the guard
                if self.retries >= 0:
                    return wrapper(*args, **kwargs)
                else:
                    raise e

        return wrapper
