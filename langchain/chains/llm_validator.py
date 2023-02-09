"""Chain to parse / validate the LLM output."""

from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE_CORRECTION = """
Output: {output}
Validator/Parser function: {validator_name}
Exception failed with error: {error_message}

---

Please correct the output so it passes the validator function.
Corrected output:
"""

PROMPT_CORRECTION = PromptTemplate(
    input_variables=["output", "validator_name", "error_message"],
    template=_PROMPT_TEMPLATE_CORRECTION,
)


class LLMCorrectChain(Chain, BaseModel):
    """Chain that corrects the input if the validator function fails."""

    llm: BaseLLM
    """LLM wrapper to use."""
    input_key: str = "text"  #: :meta private:
    output_key: str = "corrected"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        from langchain.chains.llm import LLMChain

        """Try to correct the input if the validator function fails."""
        # Small improvement: if the error message contains the whole input, remove it
        error_message = inputs["error_message"].replace(inputs["text"], "")
        prompt = PROMPT_CORRECTION.format(
            output=inputs["text"],
            validator_name=inputs["validator_name"],
            error_message=error_message,
        )
        llm_chain = LLMChain.from_string(self.llm, prompt)
        response = llm_chain(inputs)
        return {"corrected": response["text"]}

    @property
    def _chain_type(self) -> str:
        return "llm_correct_chain"
