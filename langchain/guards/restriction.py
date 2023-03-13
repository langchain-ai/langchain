"""Check if chain or agent violates one or more restrictions."""
from __future__ import annotations

from typing import Any, List, Tuple

from langchain.chains.llm import LLMChain
from langchain.guards.base import BaseGuard
from langchain.guards.restriction_prompt import RESTRICTION_PROMPT
from langchain.llms.base import BaseLLM
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts.base import BasePromptTemplate


class RestrictionGuard(BaseGuard):
    """Check if chain or agent violates one or more restrictions.

    Args:
        llm (LLM): The LLM to be used to guard the output of the chain or agent.
        restrictions (list): A list of strings that describe the restrictions that
            the output of the chain or agent must conform to. The restrictions
            should be in the form of "must not x" or "must x" for best results.
        retries (int, optional): The number of times the chain or agent should be
            called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum
            number of retries has been exceeded.

    Example:
        .. code-block:: python
            llm = OpenAI(temperature=0.9)

            text = (
                "What would be a good company name for a company"
                "that makes colorful socks? Give me a name in latin."
            )

            @RestrictionGuard(
                restrictions=['output must be in latin'], llm=llm, retries=0
            )
            def sock_idea():
                return llm(text)

            sock_idea()
    """

    def __init__(
        self,
        guard_chain: LLMChain,
        restrictions: List[str],
        retries: int = 0,
    ) -> None:
        """Initialize with restriction, prompt, and llm."""
        super().__init__(retries=retries)
        self.guard_chain = guard_chain
        self.restrictions = restrictions
        self.output_parser = BooleanOutputParser(true_values=["¥"], false_values=["ƒ"])

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        prompt: BasePromptTemplate = RESTRICTION_PROMPT,
        **kwargs: Any,
    ) -> RestrictionGuard:
        """Load from llm and prompt."""
        guard_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(guard_chain=guard_chain, **kwargs)

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Determine if guard was violated.
        Uses a custom guard chain to determine is a set of restrictions was violated.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated
                    (if guard was violated).
        """

        concatenated_restrictions = ", ".join(self.restrictions)

        # guard chain returns true in case of a violation.
        guard_chain_output = self.guard_chain.run(
            function_output=llm_response, restrictions=concatenated_restrictions
        )
        violation_message = (
            f"Restriction violated. Attempted answer: {llm_response}. "
            f"Reasoning: {guard_chain_output}."
        )
        bool_return = self.output_parser.parse(guard_chain_output)
        return bool_return, violation_message
