"""Check whe returns a large portion of a protected string (like a prompt)."""
from typing import Any, List, Tuple

from langchain.guards.base import BaseGuard


def _overlap_percent(protected_string: str, llm_response: str) -> float:
    protected_string = protected_string.lower()
    llm_response = llm_response.lower()
    len_protected, len_llm_response = len(protected_string), len(llm_response)
    max_overlap = 0
    for i in range(len_llm_response - len_protected + 1):
        for n in range(len_protected + 1):
            if llm_response[i : i + n] in protected_string:
                max_overlap = max(max_overlap, n)
    overlap_percent = max_overlap / len_protected
    return overlap_percent


class StringGuard(BaseGuard):
    """Check whe returns a large portion of a protected string (like a prompt).

    The primary use of this guard is to prevent the chain or agent from leaking
     information about its prompt or other sensitive information.
     This can also be used as a rudimentary filter of other things like profanity.

    Args:
        protected_strings (List[str]): The list of protected_strings to be guarded
        leniency (float, optional): The percentage of a protected_string that can
            be leaked before the guard is violated. Defaults to 0.5.
            For example, if the protected_string is "Tell me a joke" and the
            leniency is 0.75, then the guard will be violated if the output
            contains more than 75% of the protected_string.
            100% leniency means that the guard will only be violated when
            the string is returned exactly while 0% leniency means that the guard
            will always be violated.
        retries (int, optional): The number of times the chain or agent should be
            called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum number of retries has been exceeded.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate

            llm = OpenAI(temperature=0.9)

            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            chain = LLMChain(llm=OpenAI(), prompt=prompt)

            @StringGuard(protected_strings=[prompt], leniency=0.25 retries=1)
            def call_chain():
                return chain.run(adjective="political")

            call_chain()

    """

    def __init__(
        self, protected_strings: List[str], leniency: float = 0.5, retries: int = 0
    ) -> None:
        """Initialize with protected strings and leniency."""
        super().__init__(retries=retries)
        self.protected_strings = protected_strings
        self.leniency = leniency

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Function to determine if guard was violated.

        Checks for string leakage. Uses protected_string and leniency.
        If the output contains more than leniency * 100% of the protected_string,
         the guard is violated.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated
                    (if guard was violated).
        """

        protected_strings = self.protected_strings
        leniency = self.leniency

        for protected_string in protected_strings:
            similarity = _overlap_percent(protected_string, llm_response)
            if similarity >= leniency:
                violation_message = (
                    f"Restriction violated. Attempted answer: {llm_response}. "
                    f"Reasoning: Leakage of protected string: {protected_string}."
                )
                return True, violation_message
        return False, ""
