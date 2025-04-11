"""Combining documents by mapping a chain over them first, then reranking results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.output_parsers.regex import RegexParser


@deprecated(
    since="0.3.1",
    removal="1.0",
    message=(
        "This class is deprecated. Please see the migration guide here for "
        "a recommended replacement: "
        "https://python.langchain.com/docs/versions/migrating_chains/map_rerank_docs_chain/"  # noqa: E501
    ),
)
class MapRerankDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then reranking results.

    This algorithm calls an LLMChain on each input document. The LLMChain is expected
    to have an OutputParser that parses the result into both an answer (`answer_key`)
    and a score (`rank_key`). The answer with the highest score is then returned.

    Example:
        .. code-block:: python

            from langchain.chains import MapRerankDocumentsChain, LLMChain
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
                "you are. Context: {context}"
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
    metadata_keys: Optional[list[str]] = None
    """Additional metadata from the chosen document to return."""
    return_intermediate_steps: bool = False
    """Return intermediate steps.
    Intermediate steps include the results of calling llm_chain on each document."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        schema: dict[str, Any] = {
            self.output_key: (str, None),
        }
        if self.return_intermediate_steps:
            schema["intermediate_steps"] = (list[str], None)
        if self.metadata_keys:
            schema.update({key: (Any, None) for key in self.metadata_keys})

        return create_model("MapRerankOutput", **schema)

    @property
    def output_keys(self) -> list[str]:
        """Expect input key.

        :meta private:
        """
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = _output_keys + ["intermediate_steps"]
        if self.metadata_keys is not None:
            _output_keys += self.metadata_keys
        return _output_keys

    @model_validator(mode="after")
    def validate_llm_output(self) -> Self:
        """Validate that the combine chain outputs a dictionary."""
        output_parser = self.llm_chain.prompt.output_parser
        if not isinstance(output_parser, RegexParser):
            raise ValueError(
                "Output parser of llm_chain should be a RegexParser,"
                f" got {output_parser}"
            )
        output_keys = output_parser.output_keys
        if self.rank_key not in output_keys:
            raise ValueError(
                f"Got {self.rank_key} as key to rank on, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
        if self.answer_key not in output_keys:
            raise ValueError(
                f"Got {self.answer_key} as key to return, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """Get default document variable name, if not provided."""
        if "llm_chain" not in values:
            raise ValueError("llm_chain must be provided")

        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def combine_docs(
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
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
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
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
        docs: list[Document],
        results: Sequence[Union[str, list[str], dict[str, str]]],
    ) -> tuple[str, dict]:
        typed_results = cast(list[dict], results)
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
