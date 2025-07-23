"""Chain for interacting with Elasticsearch Database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain.chains.base import Chain
from langchain.chains.elasticsearch_database.prompts import ANSWER_PROMPT, DSL_PROMPT

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class ElasticsearchDatabaseChain(Chain):
    """Chain for interacting with Elasticsearch Database.

    Example:
        .. code-block:: python

            from langchain.chains import ElasticsearchDatabaseChain
            from langchain_community.llms import OpenAI
            from elasticsearch import Elasticsearch

            database = Elasticsearch("http://localhost:9200")
            db_chain = ElasticsearchDatabaseChain.from_llm(OpenAI(), database)
    """

    query_chain: Runnable
    """Chain for creating the ES query."""
    answer_chain: Runnable
    """Chain for answering the user question."""
    database: Any = None
    """Elasticsearch database to connect to of type elasticsearch.Elasticsearch."""
    top_k: int = 10
    """Number of results to return from the query"""
    ignore_indices: Optional[list[str]] = None
    include_indices: Optional[list[str]] = None
    input_key: str = "question"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    sample_documents_in_index_info: int = 3
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        if self.include_indices and self.ignore_indices:
            msg = "Cannot specify both 'include_indices' and 'ignore_indices'."
            raise ValueError(msg)
        return self

    @property
    def input_keys(self) -> list[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        return [self.output_key, INTERMEDIATE_STEPS_KEY]

    def _list_indices(self) -> list[str]:
        all_indices = [
            index["index"] for index in self.database.cat.indices(format="json")
        ]

        if self.include_indices:
            all_indices = [i for i in all_indices if i in self.include_indices]
        if self.ignore_indices:
            all_indices = [i for i in all_indices if i not in self.ignore_indices]

        return all_indices

    def _get_indices_infos(self, indices: list[str]) -> str:
        mappings = self.database.indices.get_mapping(index=",".join(indices))
        if self.sample_documents_in_index_info > 0:
            for k, v in mappings.items():
                hits = self.database.search(
                    index=k,
                    query={"match_all": {}},
                    size=self.sample_documents_in_index_info,
                )["hits"]["hits"]
                hits = [str(hit["_source"]) for hit in hits]
                mappings[k]["mappings"] = str(v) + "\n\n/*\n" + "\n".join(hits) + "\n*/"
        return "\n\n".join(
            [
                "Mapping for index {}:\n{}".format(index, mappings[index]["mappings"])
                for index in mappings
            ],
        )

    def _search(self, indices: list[str], query: str) -> str:
        result = self.database.search(index=",".join(indices), body=query)
        return str(result)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nESQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        indices = self._list_indices()
        indices_info = self._get_indices_infos(indices)
        query_inputs: dict = {
            "input": input_text,
            "top_k": str(self.top_k),
            "indices_info": indices_info,
            "stop": ["\nESResult:"],
        }
        intermediate_steps: list = []
        try:
            intermediate_steps.append(query_inputs)  # input: es generation
            es_cmd = self.query_chain.invoke(
                query_inputs,
                config={"callbacks": _run_manager.get_child()},
            )

            _run_manager.on_text(es_cmd, color="green", verbose=self.verbose)
            intermediate_steps.append(
                es_cmd,
            )  # output: elasticsearch dsl generation (no checker)
            intermediate_steps.append({"es_cmd": es_cmd})  # input: ES search
            result = self._search(indices=indices, query=es_cmd)
            intermediate_steps.append(str(result))  # output: ES search

            _run_manager.on_text("\nESResult: ", verbose=self.verbose)
            _run_manager.on_text(result, color="yellow", verbose=self.verbose)

            _run_manager.on_text("\nAnswer:", verbose=self.verbose)
            answer_inputs: dict = {"data": result, "input": input_text}
            intermediate_steps.append(answer_inputs)  # input: final answer
            final_result = self.answer_chain.invoke(
                answer_inputs,
                config={"callbacks": _run_manager.get_child()},
            )

            intermediate_steps.append(final_result)  # output: final answer
            _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore[attr-defined]
            raise

        return chain_result

    @property
    def _chain_type(self) -> str:
        return "elasticsearch_database_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database: Elasticsearch,
        *,
        query_prompt: Optional[BasePromptTemplate] = None,
        answer_prompt: Optional[BasePromptTemplate] = None,
        query_output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> ElasticsearchDatabaseChain:
        """Convenience method to construct ElasticsearchDatabaseChain from an LLM.

        Args:
            llm: The language model to use.
            database: The Elasticsearch db.
            query_prompt: The prompt to use for query construction.
            answer_prompt: The prompt to use for answering user question given data.
            query_output_parser: The output parser to use for parsing model-generated
                ES query. Defaults to SimpleJsonOutputParser.
            kwargs: Additional arguments to pass to the constructor.
        """
        query_prompt = query_prompt or DSL_PROMPT
        query_output_parser = query_output_parser or SimpleJsonOutputParser()
        query_chain = query_prompt | llm | query_output_parser
        answer_prompt = answer_prompt or ANSWER_PROMPT
        answer_chain = answer_prompt | llm | StrOutputParser()
        return cls(
            query_chain=query_chain,
            answer_chain=answer_chain,
            database=database,
            **kwargs,
        )
