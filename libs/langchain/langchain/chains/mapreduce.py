"""Map-reduce chain.

Splits up a document, sends the smaller parts to the LLM with one prompt,
then combines the results with another one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun, Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_text_splitters import TextSplitter
from pydantic import ConfigDict

from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "Refer to migration guide here for a recommended implementation using "
        "LangGraph: https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/"  # noqa: E501
        ". See also LangGraph guides for map-reduce: "
        "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/."
    ),
)
class MapReduceChain(Chain):
    """Map-reduce chain."""

    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine documents."""
    text_splitter: TextSplitter
    """Text splitter to use."""
    input_key: str = "input_text"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

    @classmethod
    def from_params(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        text_splitter: TextSplitter,
        callbacks: Callbacks = None,
        combine_chain_kwargs: Optional[Mapping[str, Any]] = None,
        reduce_chain_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> MapReduceChain:
        """Construct a map-reduce chain that uses the chain for map and reduce."""
        llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            callbacks=callbacks,
            **(reduce_chain_kwargs if reduce_chain_kwargs else {}),
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=stuff_chain
        )
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
            callbacks=callbacks,
            **(combine_chain_kwargs if combine_chain_kwargs else {}),
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            text_splitter=text_splitter,
            callbacks=callbacks,
            **kwargs,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # Split the larger text into smaller chunks.
        doc_text = inputs.pop(self.input_key)
        texts = self.text_splitter.split_text(doc_text)
        docs = [Document(page_content=text) for text in texts]
        _inputs: Dict[str, Any] = {
            **inputs,
            self.combine_documents_chain.input_key: docs,
        }
        outputs = self.combine_documents_chain.run(
            _inputs, callbacks=_run_manager.get_child()
        )
        return {self.output_key: outputs}
