from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_generation.prompt import PROMPT_SELECTOR


@deprecated(
    since="0.2.7",
    alternative=(
        "example in API reference with more detail: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_generation.base.QAGenerationChain.html"  # noqa: E501
    ),
    removal="1.0",
)
class QAGenerationChain(Chain):
    """Base class for question-answer generation chains.

    This class is deprecated. See below for an alternative implementation.

    Advantages of this implementation include:

    - Supports async and streaming;
    - Surfaces prompt and text splitter for easier customization;
    - Use of JsonOutputParser supports JSONPatch operations in streaming mode,
      as well as robustness to markdown.

        .. code-block:: python

            from langchain.chains.qa_generation.prompt import CHAT_PROMPT as prompt
            # Note: import PROMPT if using a legacy non-chat model.
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.runnables import (
                RunnableLambda,
                RunnableParallel,
                RunnablePassthrough,
            )
            from langchain_core.runnables.base import RunnableEach
            from langchain_openai import ChatOpenAI
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            llm = ChatOpenAI()
            text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500)
            split_text = RunnableLambda(
                lambda x: text_splitter.create_documents([x])
            )

            chain = RunnableParallel(
                text=RunnablePassthrough(),
                questions=(
                    split_text | RunnableEach(bound=prompt | llm | JsonOutputParser())
                )
            )
    """

    llm_chain: LLMChain
    """LLM Chain that generates responses from user input and context."""
    text_splitter: TextSplitter = Field(
        default=RecursiveCharacterTextSplitter(chunk_overlap=500)
    )
    """Text splitter that splits the input into chunks."""
    input_key: str = "text"
    """Key of the input to the chain."""
    output_key: str = "questions"
    """Key of the output of the chain."""
    k: Optional[int] = None
    """Number of questions to generate."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> QAGenerationChain:
        """
        Create a QAGenerationChain from a language model.

        Args:
            llm: a language model
            prompt: a prompt template
            **kwargs: additional arguments

        Returns:
            a QAGenerationChain class
        """
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])
        results = self.llm_chain.generate(
            [{"text": d.page_content} for d in docs], run_manager=run_manager
        )
        qa = [json.loads(res[0].text) for res in results.generations]
        return {self.output_key: qa}
