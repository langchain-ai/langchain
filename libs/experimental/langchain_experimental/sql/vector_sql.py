"""Vector SQL Database Chain Retriever"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.prompts.prompt import PromptTemplate
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

from langchain_experimental.sql.base import INTERMEDIATE_STEPS_KEY, SQLDatabaseChain


class VectorSQLOutputParser(BaseOutputParser[str]):
    """Output Parser for Vector SQL.

    1. finds for `NeuralArray()` and replace it with the embedding
    2. finds for `DISTANCE()` and replace it with the distance name in backend SQL
    """

    model: Embeddings
    """Embedding model to extract embedding for entity"""
    distance_func_name: str = "distance"
    """Distance name for Vector SQL"""

    class Config:
        arbitrary_types_allowed = 1

    @property
    def _type(self) -> str:
        return "vector_sql_parser"

    @classmethod
    def from_embeddings(
        cls, model: Embeddings, distance_func_name: str = "distance", **kwargs: Any
    ) -> BaseOutputParser:
        return cls(model=model, distance_func_name=distance_func_name, **kwargs)

    def parse(self, text: str) -> str:
        text = text.strip()
        start = text.find("NeuralArray(")
        _sql_str_compl = text
        if start > 0:
            _matched = text[text.find("NeuralArray(") + len("NeuralArray(") :]
            end = _matched.find(")") + start + len("NeuralArray(") + 1
            entity = _matched[: _matched.find(")")]
            vecs = self.model.embed_query(entity)
            vecs_str = "[" + ",".join(map(str, vecs)) + "]"
            _sql_str_compl = text.replace("DISTANCE", self.distance_func_name).replace(
                text[start:end], vecs_str
            )
            if _sql_str_compl[-1] == ";":
                _sql_str_compl = _sql_str_compl[:-1]
        return _sql_str_compl


class VectorSQLRetrieveAllOutputParser(VectorSQLOutputParser):
    """Parser based on VectorSQLOutputParser.
    It also modifies the SQL to get all columns.
    """

    @property
    def _type(self) -> str:
        return "vector_sql_retrieve_all_parser"

    def parse(self, text: str) -> str:
        text = text.strip()
        start = text.upper().find("SELECT")
        if start >= 0:
            end = text.upper().find("FROM")
            text = text.replace(text[start + len("SELECT") + 1 : end - 1], "*")
        return super().parse(text)


def get_result_from_sqldb(db: SQLDatabase, cmd: str) -> Sequence[Dict[str, Any]]:
    """Get result from SQL Database."""

    result = db._execute(cmd, fetch="all")
    assert isinstance(result, Sequence)
    return result


class VectorSQLDatabaseChain(SQLDatabaseChain):
    """Chain for interacting with Vector SQL Database.

    Example:
        .. code-block:: python

            from langchain_experimental.sql import SQLDatabaseChain
            from langchain_community.llms import OpenAI, SQLDatabase, OpenAIEmbeddings
            db = SQLDatabase(...)
            db_chain = VectorSQLDatabaseChain.from_llm(OpenAI(), db, OpenAIEmbeddings())

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include the permissions this chain needs.
        Failure to do so may result in data corruption or loss, since this chain may
        attempt commands like `DROP TABLE` or `INSERT` if appropriately prompted.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this chain.
        This issue shows an example negative outcome if these steps are not taken:
        https://github.com/langchain-ai/langchain/issues/5923
    """

    sql_cmd_parser: VectorSQLOutputParser
    """Parser for Vector SQL"""
    native_format: bool = False
    """If return_direct, controls whether to return in python native format"""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)  # input: sql generation
            llm_out = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            )
            sql_cmd = self.sql_cmd_parser.parse(llm_out)
            if self.return_sql:
                return {self.output_key: sql_cmd}
            if not self.use_query_checker:
                _run_manager.on_text(llm_out, color="green", verbose=self.verbose)
                intermediate_steps.append(
                    llm_out
                )  # output: sql generation (no checker)
                intermediate_steps.append({"sql_cmd": llm_out})  # input: sql exec
                result = get_result_from_sqldb(self.database, sql_cmd)
                intermediate_steps.append(str(result))  # output: sql exec
            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm_chain.llm,
                    prompt=query_checker_prompt,
                    output_parser=self.llm_chain.output_parser,
                )
                query_checker_inputs = {
                    "query": llm_out,
                    "dialect": self.database.dialect,
                }
                checked_llm_out = query_checker_chain.predict(
                    callbacks=_run_manager.get_child(), **query_checker_inputs
                )
                checked_sql_command = self.sql_cmd_parser.parse(checked_llm_out)
                intermediate_steps.append(
                    checked_llm_out
                )  # output: sql generation (checker)
                _run_manager.on_text(
                    checked_llm_out, color="green", verbose=self.verbose
                )
                intermediate_steps.append(
                    {"sql_cmd": checked_llm_out}
                )  # input: sql exec
                result = get_result_from_sqldb(self.database, checked_sql_command)
                intermediate_steps.append(str(result))  # output: sql exec
                llm_out = checked_llm_out
                sql_cmd = checked_sql_command

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(str(result), color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result (`Sequence[Dict[str, Any]]`),
            # otherwise try to get a human readable final answer (`str`).
            final_result: Union[str, Sequence[Dict[str, Any]]]
            if self.return_direct:
                final_result = result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{llm_out}\nSQLResult: {result}\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs)  # input: final answer
                final_result = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()
                intermediate_steps.append(final_result)  # output: final answer
                _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc

    @property
    def _chain_type(self) -> str:
        return "vector_sql_database_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        sql_cmd_parser: Optional[VectorSQLOutputParser] = None,
        **kwargs: Any,
    ) -> VectorSQLDatabaseChain:
        assert sql_cmd_parser, "`sql_cmd_parser` must be set in VectorSQLDatabaseChain."
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            llm_chain=llm_chain, database=db, sql_cmd_parser=sql_cmd_parser, **kwargs
        )
