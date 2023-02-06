"""A utilities for guarding the output of a given chain or agent.
"""

from typing import Any, Callable, List, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities.normalization import normalize_boolean_output

from difflib import SequenceMatcher


class BaseGuard:
    """The Guard class is a decorator that can be applied to any chain or agent to either throw an error or recursively call the chain or agent when the output of said chain or agent violates the rules of the guard. The BaseGuard alone does nothing but can be subclassed and the resolve_guard function overwritten to create more specific guards.

    Args:
        retries (int, optional): The number of times the chain or agent should be called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum number of retries has been exceeded.

    """

    def __init__(self, retries: int = 0, *args: Any, **kwargs: Any) -> None:
        self.retries = retries

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Function to determine if guard was violated (if response should be blocked)
        Can be overwritten when subclassing Guard class to expand on guard functionality

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated (if guard was violated).
        """
        return False, ""

    def handle_violation(self, message: str, *args: Any, **kwargs: Any) -> Exception:
        """Function to handle violation of guard.

        Args:
            message (str): the message to be displayed when the guard is violated.

        Raises:
            Exception: the message passed to the function.
        """
        raise Exception(message)

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> str:
            if self.retries < 0:
                raise Exception("Restriction violated. Maximum retries exceeded.")
            try:
                llm_response = func(*args, **kwargs)
                guard_result, violation_message = self.resolve_guard(llm_response)
                if guard_result:
                    self.handle_violation(violation_message)
                else:
                    return llm_response
            except Exception as e:
                self.retries = self.retries - 1
                # Check retries to avoid infinite recursion if exception is something other than a violation of the guard
                if self.retries >= 0:
                    return wrapper(*args, **kwargs)
                else:
                    raise e

        return wrapper


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
            normalize_boolean_output(
                guard_chain_output, true_values=["¥"], false_values=["ƒ"]
            ),
            violation_message,
        )


class CustomGuard(BaseGuard):
    """The CustomGuard class is a decorator that can be applied to any chain or agent to either throw an error or recursively call the chain or agent when the output of said chain or agent violates a provided guard function

    Args:
        guard_function (func): The function to be used to guard the output of the chain or agent. The function should take the output of the chain or agent as its only argument and return a boolean value where True means the guard has been violated. Optionally, return a tuple where the first element is a boolean value and the second element is a string that will be displayed when the guard is violated. If the string is ommited the default message will be used.
        retries (int, optional): The number of times the chain or agent should be called recursively if the output violates the restrictions. Defaults to 0.

    Raises:
        Exception: If the output violates the restrictions and the maximum number of retries has been exceeded.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            from langchain.utilities.guards import CustomGuard

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
        self.guard_function = guard_function
        self.retries = retries

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Function to determine if guard was violated. Uses custom guard function.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated (if guard was violated).
        """
        response = self.guard_function(llm_response)

        if type(response) is tuple:
            boolean_output, message = response
            violation_message = message
        elif type(response) is bool:
            boolean_output = response
            violation_message = (
                "Restriction violated. Attempted answer: " + llm_response + "."
            )
        else:
            raise Exception(
                "Custom guard function must return either a boolean or a tuple of a boolean and a string."
            )
        return (boolean_output, violation_message)


class StringGuard(BaseGuard):
    """The StringGuard class is a decorator that can be applied to any chain or agent to either throw an error or recursively call the chain or agent when the output of said chain or agent returns a large portion of a protected string (like a prompt.) The primary use of this guard is to prevent the chain or agent from leaking information about its prompt or other sensitive information. This can also be used as a rudimentary filter of other things like profanity, though.

    Args:
        protected_strings (List[str]): The list of protected_strings to be guarded
        leniency (float, optional): The percentage of a protected_string that can be leaked before the guard is violated. Defaults to 0.5. For example, if the protected_string is "Tell me a joke" and the leniency is 0.75, then the guard will be violated if the output contains more than 75% of the protected_string. 100% leniency means that the guard will only be violated when the string is returned exactly while 0% leniency means that the guard will always be violated.
        retries (int, optional): The number of times the chain or agent should be called recursively if the output violates the restrictions. Defaults to 0.

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
        self.protected_strings = protected_strings
        self.leniency = leniency
        self.retries = retries

    def resolve_guard(
        self, llm_response: str, *args: Any, **kwargs: Any
    ) -> Tuple[bool, str]:
        """Function to determine if guard was violated. Checks for string leakage. Uses protected_string and leniency. If the output contains more than leniency * 100% of the protected_string, the guard is violated.

        Args:
            llm_response (str): the llm_response string to be tested against the guard.

        Returns:
            tuple:
                bool: True if guard was violated, False otherwise.
                str: The message to be displayed when the guard is violated (if guard was violated).
        """

        def overlap_percent(protected_string, llm_response):
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

        protected_strings = self.protected_strings
        leniency = self.leniency

        for protected_string in protected_strings:
            similarity = overlap_percent(protected_string, llm_response)
            print(protected_string)
            print(similarity)
            if similarity >= leniency:
                violation_message = (
                    "Restriction violated. Attempted answer: "
                    + llm_response
                    + ". Reasoning: "
                    + "Leakage of protected string: "
                    + protected_string
                    + "."
                )
                return (True, violation_message)
        return (False, "")
