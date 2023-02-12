"""Check if chain or agent violates a provided guard function."""
from typing import Any, Callable, Tuple

from langchain.guards.base import BaseGuard


class CustomGuard(BaseGuard):
    """Check if chain or agent violates a provided guard function.

    Args:
        guard_function (func): The function to be used to guard the
            output of the chain or agent. The function should take
            the output of the chain or agent as its only argument
            and return a boolean value where True means the guard
            has been violated. Optionally, return a tuple where the
            first element is a boolean value and the second element is
            a string that will be displayed when the guard is violated.
            If the string is ommited the default message will be used.
        retries (int, optional): The number of times the chain or agent
            should be called recursively if the output violates the
            restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the
            maximum number of retries has been exceeded.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            from langchain.guards import CustomGuard

            llm = OpenAI(temperature=0.9)

            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            chain = LLMChain(llm=OpenAI(), prompt=prompt)

            def is_long(llm_output):
                return len(llm_output) > 100

            @CustomGuard(guard_function=is_long, retries=1)
            def call_chain():
                return chain.run(adjective="political")

            call_chain()

    """

    def __init__(self, guard_function: Callable, retries: int = 0) -> None:
        """Initialize with guard function and retries."""
        super().__init__(retries=retries)
        self.guard_function = guard_function

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Determine if guard was violated. Uses custom guard function.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated
                    (if guard was violated).
        """
        response = self.guard_function(llm_response)

        if type(response) is tuple:
            boolean_output, message = response
            violation_message = message
        elif type(response) is bool:
            boolean_output = response
            violation_message = (
                f"Restriction violated. Attempted answer: {llm_response}."
            )
        else:
            raise Exception(
                "Custom guard function must return either a boolean"
                " or a tuple of a boolean and a string."
            )
        return boolean_output, violation_message
