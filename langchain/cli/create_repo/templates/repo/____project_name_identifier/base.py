"""This is a template for a custom chain.

Edit this file to implement your chain logic.
"""

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.chains.base import Chain

MY_PROMPT = """You are an assistant that helps to form nice and human understandable answers.

Information:
{my_template_var}

Question: {question}
Helpful Answer:"""  # noqa: E501


class MyChain(Chain):
    """One-line description of my chain.

    Example:
        .. code-block:: python

            from langchain.chains import LLMChain
            from langchain.llms import OpenAI
            from langchain.prompts import PromptTemplate

            from ____project_name_identifier import MyChain

            prompt = PromptTemplate.from_template("foo {bar}")
            llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)
            my_chain = MyChain(llm_chain=llm_chain)
            my_chain.run(bar="baz")
    """

    llm_chain: LLMChain
    """A description of each argument to my chain."""

    @property
    def input_keys(self) -> List[str]:
        """Input variables to my chain."""
        return self.llm_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output keys of my chain."""
        return self.llm_chain.output_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Execute the chain.

        This is a private method that is not user-facing. It is only called within
            `Chain.__call__`, which is the user-facing wrapper method that handles
            callbacks configuration and some input/output processing.

        Args:
            inputs: A dict of named inputs to the chain. Assumed to contain all inputs
                specified in `Chain.input_keys`, including any inputs added by memory.
            run_manager: The callbacks manager that contains the callback handlers for
                this run of the chain.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """
        return self.llm_chain(
            inputs, callbacks=run_manager.get_child() if run_manager else None
        )
