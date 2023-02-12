from typing import Any, Callable, List, Tuple

from langchain import LLMChain, PromptTemplate
from langchain.guards.base import BaseGuard
from langchain.output_parsing.boolean import BooleanOutputParser


class RestrictionGuard(BaseGuard):
    """The RestrictionGuard class is a decorator that can be applied to any chain or agent to either throw an error or recursively call the chain or agent when the output of said chain or agent violates one or more restrictions.

    Args:
        llm (LLM): The LLM to be used to guard the output of the chain or agent.
        restrictions (list): A list of strings that describe the restrictions that the output of the chain or agent must conform to. The restrictions should be in the form of "must not x" or "must x" for best results.
        retries (int, optional): The number of times the chain or agent should be called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum number of retries has been exceeded.

    Example:
        .. code-block:: python
            llm = OpenAI(temperature=0.9)

            text = "What would be a good company name for a company that makes colorful socks? Give me a name in latin."

            @RestrictionGuard(restrictions=['output must be in latin'], llm=llm, retries=0)
            def sock_idea():
                return llm(text)

            sock_idea()
    """

    prompt = PromptTemplate(
        input_variables=["function_output", "restrictions"],
        template="""Determine if the output of a function violates any restrictions. If the output violates any of the restrictions, respond with your reason followed by ¥. If the output does not violate any restrictions, respond with your reason followed by ƒ.

    output: Lorum ipsum dolor sit amet
    restrictions: must not include profanity
    response: output does not include profanity (ƒ)

    output: Lorum ipsum dolor sit amet
    restrictions: must not include latin
    response: output includes latin (¥)

    output: I hate you, you ugly jerk!
    restrictions: must not be negative, must not be an insult
    response: output is negative and an insult (¥)

    output: {function_output}
    restrictions:{restrictions}
    response:
    """,
        # TODO: Should probably replace some of the n-shot examples with more realistic ones about race and profanity. Not sure about the idea of introducing racist and profane sentences into the repo for that purpose?
    )

    def __init__(
        self, llm: Callable, restrictions: List[str], retries: int = 0
    ) -> None:
        self.llm = llm
        self.restrictions = restrictions
        self.retries = retries
        self.output_parser = BooleanOutputParser(true_values=["¥"], false_values=["ƒ"])

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Function to determine if guard was violated. Uses a custom guard chain to determine is a set of restrictions was violated.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated (if guard was violated).
        """
        guard_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        concatenated_restrictions = ", ".join(self.restrictions)

        # guard chain returns true in case of a violation.
        guard_chain_output = guard_chain.run(
            function_output=llm_response, restrictions=concatenated_restrictions
        )
        violation_message = (
            "Restriction violated. Attempted answer: "
            + llm_response
            + ". Reasoning: "
            + guard_chain_output
            + "."
        )
        # rare characters are used for normalization so that an explanation can be included but easily be filtered out without worrying about the explanation containing the same characters as the true or false values.
        return (
            self.output_parser.parse(
                guard_chain_output,
            ),
            violation_message,
        )
