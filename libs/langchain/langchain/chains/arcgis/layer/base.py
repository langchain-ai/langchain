from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.arcgis.layer.prompts import PROMPT
from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.document_transformers.arcgis_row_summarizer import (
    ArcGISRowSummaryTransformer,
)
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import Extra, Field
from langchain.schema import (
    Document,
)


class ArcGISLayerSummaryInnerChain(LLMChain):
    """
    Represents a custom chain to generate overall layer
    summaries for geospatial data layers using an LLM.

    Attributes:
        llm (BaseChatModel): The Large Language Model used for text generation.
        prompt (ChatPromptTemplate): The template to guide the LLM's responses.
        output_key (str): The key used to extract the output from the LLM's response.
    """

    llm: BaseChatModel
    prompt: ChatPromptTemplate = PROMPT
    output_key: str = "text"

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)

        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # if run_manager:
        #     run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)

        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # if run_manager:
        #     await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "ArcGISLayerSummaryInnerChain"


def _get_default_document_prompt() -> ChatPromptTemplate:
    return PROMPT


class ArcGISLayerStuffSummaryChain(BaseCombineDocumentsChain):
    llm_chain: ArcGISLayerSummaryInnerChain
    """LLM chain which is called with the formatted document string,
    along with any other inputs."""
    document_prompt: ChatPromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @staticmethod
    def summaries_str(docs: Sequence[Document]) -> str:
        """
        Converts a list of summaries into a formatted string.

        Args:
            summaries (list[str]): A list of summaries.

        Returns:
            str: A formatted string containing the summaries.
        """
        summaries = (doc.page_content for doc in docs)
        sub_sums = "\n".join(
            f"<row_summary>\n{summary}\n</row_summary>" for summary in summaries
        )
        return f"<layer_summary>\n{sub_sums}\n</layer_summary>"

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        inputs = dict(
            name=docs[0].metadata["name"],
            desc=ArcGISRowSummaryTransformer.desc_from_doc(docs[0]),
            summaries_str=self.summaries_str(docs),
        )
        return inputs

    def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain.llm.get_num_tokens(prompt)

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return self.llm_chain.predict(callbacks=callbacks, **inputs), {}

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Async stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return await self.llm_chain.apredict(callbacks=callbacks, **inputs), {}

    @property
    def _chain_type(self) -> str:
        return "ArcGISLayerStuffSummaryChain"
