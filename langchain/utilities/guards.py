"""A utilities for guarding the output of a given chain or agent.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.utilities.normalization import normalize_boolean_output


class Guard:
    """The Guard class is a decorator that can be applied to any chain or agent to either throw an error or recursively call the chain or agent when the output of said chain or agent conforms to one or more restrictions.

    Args:
        llm (LLM): The LLM to be used to guard the output of the chain or agent.
        restrictions (list): A list of strings that describe the restrictions that the output of the chain or agent must conform to.
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

            @Guard(llm=llm, restrictions=['must not include profanity', 'must not mention race.'], retries=1)
            def call_chain():
                return chain.run(adjective="political")

            call_chain()

    """

    prompt = PromptTemplate(
    input_variables=["function_output", "restrictions"],
    template="""Determine if the output of a function violates any restrictions. If the output violates any of the restrictions, respond with your reason followed by ¥. If the output does not violate any restrictions, respond with your reason followed by ƒ. If you are unsure, respond with your reason followed by ƒ.

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

    def __init__(self, llm, restrictions, retries=0):
        self.llm = llm
        self.restrictions = restrictions
        self.retries = retries

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            guard_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            if self.retries < 0:
                raise Exception("Restriction violated. Maximum retries exceeded.")
            try:
                concatenated_restrictions = ', '.join(self.restrictions)
                result = func(*args, **kwargs)
                # guard chain returns true in case of a violation.
                guard_chain_output = guard_chain.run(function_output=result, restrictions=concatenated_restrictions)
                if normalize_boolean_output(guard_chain_output, true_values=["¥"], false_values=["ƒ"]):
                    raise Exception("Restriction violated. Attempted answer: " + result + ". Reasoning: " + guard_chain_output + "." )
                else:
                    return result
            except Exception as e:
                # if template is in args add restriction to the end of the template
                if "template" in kwargs:
                    kwargs["template"] = kwargs["template"] + f"the response {self.restrictions[0]}"
                self.retries = self.retries - 1
                return wrapper(*args, **kwargs)
        return wrapper


