"""Chain for wrapping a retriever."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever, Document


class MlflowRetrieverEvaluator(Chain):
    input_key: str = "query"  #: :meta private:
    output_key: str = "source_documents"  #: :meta private:
    retriever: BaseRetriever = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        return [self.output_key]

    def _get_docs(self, question: str) -> List[Document]:
        """Get documents from the retriever."""
        return self.retriever.get_relevant_documents(question)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run _get_docs on input query.

        Returns the retrieved documents under the key 'source_documents'.

        Example:
        .. code-block:: python

        chain = MlflowRetrieverEvaluator(retriever=...)
        res = chain({'query': 'This is my query'})
        docs = res['source_documents']
        """
        question = inputs[self.input_key]
        docs = self._get_docs(question)
        return {self.output_key: docs}

    async def _aget_docs(self, question: str) -> List[Document]:
        """Get documents from the retriever."""
        return await self.retriever.aget_relevant_documents(question)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run _get_docs on input query.

        Returns the retrieved documents under the key 'source_documents'.

        Example:
        .. code-block:: python

        chain = MlflowRetrieverEvaluator(retriever=...)
        res = chain({'query': 'This is my query'})
        docs = res['source_documents']
        """
        question = inputs[self.input_key]
        docs = await self._aget_docs(question)
        return {self.output_key: docs}

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "mlflow_retriever_evaluator"
