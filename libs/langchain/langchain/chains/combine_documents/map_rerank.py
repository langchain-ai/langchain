"""Combining documents by mapping a chain over them first, then reranking results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.output_parsers.regex import RegexParser


class MapRerankDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then reranking results.

    This algorithm calls an LLMChain on each input document. The LLMChain is expected
    to have an OutputParser that parses the result into both an answer (`answer_key`)
    and a score (`rank_key`). The answer with the highest score is then returned.

    Example:
        .. code-block:: python

            from langchain.chains import StuffDocumentsChain, LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI
            from langchain.output_parsers.regex import RegexParser

            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            # The actual prompt will need to be a lot more complex, this is just
            # an example.
            prompt_template = (
                "Use the following context to tell me the chemical formula "
                "for water. Output both your answer and a score of how confident "
                "you are. Context: {content}"
            )
            output_parser = RegexParser(
                regex=r"(.*?)\nScore: (.*)",
                output_keys=["answer", "score"],
            )
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"],
                output_parser=output_parser,
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            chain = MapRerankDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name=document_variable_name,
                rank_key="score",
                answer_key="answer",
            )
    """

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    rank_key: str
    """Key in output of llm_chain to rank on."""
    answer_key: str
    """Key in output of llm_chain to return as answer."""
    metadata_keys: Optional[List[str]] = None
    """Additional metadata from the chosen document to return."""
    return_intermediate_steps: bool = False
    """Return intermediate steps.
    Intermediate steps include the results of calling llm_chain on each document."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        schema: Dict[str, Any] = {
            self.output_key: (str, None),
        }
        if self.return_intermediate_steps:
            schema["intermediate_steps"] = (List[str], None)
        if self.metadata_keys:
            schema.update({key: (Any, None) for key in self.metadata_keys})

        return create_model("MapRerankOutput", **schema)

    @property
    def output_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = _output_keys + ["intermediate_steps"]
        if self.metadata_keys is not None:
            _output_keys += self.metadata_keys
        return _output_keys

    @root_validator()
    def validate_llm_output(cls, values: Dict) -> Dict:
        """Validate that the combine chain outputs a dictionary."""
        output_parser = values["llm_chain"].prompt.output_parser
        if not isinstance(output_parser, RegexParser):
            raise ValueError(
                "Output parser of llm_chain should be a RegexParser,"
                f" got {output_parser}"
            )
        output_keys = output_parser.output_keys
        if values["rank_key"] not in output_keys:
            raise ValueError(
                f"Got {values['rank_key']} as key to rank on, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
        if values["answer_key"] not in output_keys:
            raise ValueError(
                f"Got {values['answer_key']} as key to return, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
        return values

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided."""
        if "document_variable_name" not in values:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents in a map rerank manner.

        Combine by mapping first chain over all documents, then reranking the results.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        results = self.llm_chain.apply_and_parse(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        return self._process_results(docs, results)

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents in a map rerank manner.

        Combine by mapping first chain over all documents, then reranking the results.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        results = await self.llm_chain.aapply_and_parse(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        return self._process_results(docs, results)

    def _process_results(
        self,
        docs: List[Document],
        results: Sequence[Union[str, List[str], Dict[str, str]]],
    ) -> Tuple[str, dict]:
        typed_results = cast(List[dict], results)
        sorted_res = sorted(
            zip(typed_results, docs), key=lambda x: -int(x[0][self.rank_key])
        )
        output, document = sorted_res[0]
        extra_info = {}
        if self.metadata_keys is not None:
            for key in self.metadata_keys:
                extra_info[key] = document.metadata[key]
        if self.return_intermediate_steps:
            extra_info["intermediate_steps"] = results
        return output[self.answer_key], extra_info

    @property
    def _chain_type(self) -> str:
        return "map_rerank_documents_chain"
