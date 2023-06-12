"""Chain for interacting with SQL Database."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import sqlfluff
from pydantic import Extra, Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.tools.sql_database.prompt import QUERY_CHECKER

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def get_json_segment(
    parse_result: Dict[str, Any], segment_type: str
) -> Iterator[Union[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Recursively search JSON parse result for specified segment type.

    Args:
        parse_result (Dict[str, Any]): JSON parse result from `sqlfluff.fix`.
        segment_type (str): The segment type to search for.

    Yields:
        Iterator[Union[str, Dict[str, Any], List[Dict[str, Any]]]]:
        Retrieves children of specified segment type as either a string for a raw
        segment or as JSON or an array of JSON for non-raw segments.
    """
    for k, v in parse_result.items():
        if k == segment_type:
            yield v
        elif isinstance(v, dict):
            yield from get_json_segment(v, segment_type)
        elif isinstance(v, list):
            for s in v:
                yield from get_json_segment(s, segment_type)


def validate_sql(sql_cmd: str, dialect: str, sql_validation: SQLValidation) -> None:
    """Parse an SQL query from a given dialect and determine if it passes validations.

    Args:
        sql_cmd (str): SQL query to validate.
        dialect (str): Dialect of the SQL query.
        sql_validation (SQLValidation): Determines which validations need to be performed

    """
    try:
        parse_result = sqlfluff.parse(sql=sql_cmd, dialect=dialect)
    except sqlfluff.api.simple.APIParsingError as e:
        raise ValueError(f"Parsing of SQL query `{sql_cmd}` failed: {e.msg}")
    except sqlfluff.core.errors.SQLFluffUserError as e:
        unsupported_dialect_message = f"Dialect {dialect} unsupported for SQL validation. No validation will be done. Go to https://docs.sqlfluff.com/en/stable/dialects.html to see supported dialects"
        if sql_validation.allow_unsupported_dialect is False:
            raise ValueError(unsupported_dialect_message)
        warnings.warn(unsupported_dialect_message)
        return

    if sql_validation.allow_non_select_statements is False:
        statements = get_json_segment(parse_result, "statement")
        for statement in statements:
            if hasattr(statement, "keys") and list(statement.keys())[0] != "select_statement":
                raise ValueError(
                    f"Found disallowed non select statement `{statement}` in SQL query `{sql_cmd}`"
                )

    if sql_validation.allow_select_all_statements is False:
        select_statements = get_json_segment(parse_result, "select_statement")
        for select_statement in select_statements:
            if not isinstance(select_statement, dict):
                continue
            wildcard_expressions = get_json_segment(
                select_statement, "wildcard_expression"
            )
            found_wildcard_expressions = list(wildcard_expressions)
            if len(found_wildcard_expressions) > 0:
                raise ValueError(
                    f"Found disallowed wildcard select statement(s) `{found_wildcard_expressions}` in query `{sql_cmd}`"
                )


class SQLValidation(object):
    def __init__(
        self,
        allow_unsupported_dialect: bool = True,
        allow_non_select_statements: bool = False,
        allow_select_all_statements: bool = True,
    ):
        """Initialize an SQLValidation instance

        Args:
            allow_unsupported_dialect (bool): If unsupported dialects are allowed, no validations will be done on them.
            allow_non_select_statements (bool): Allow statments that are non select ones, such as DROP.
            allow_select_all_statements (bool): Allow statments that are selecting all columns (SELECT *).
        """
        self.allow_unsupported_dialect = allow_unsupported_dialect
        self.allow_non_select_statements = allow_non_select_statements
        self.allow_select_all_statements = allow_select_all_statements


class SQLDatabaseChain(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain.from_llm(OpenAI(), db)
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 5
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""
    use_query_checker: bool = False
    """Whether or not the query checker tool should be used to attempt 
    to fix the initial SQL from the LLM."""
    query_checker_prompt: Optional[BasePromptTemplate] = None
    """The prompt template that should be used by the query checker"""
    sql_validation: SQLValidation = SQLValidation()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an SQLDatabaseChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                database = values["database"]
                prompt = values.get("prompt") or SQL_PROMPTS.get(
                    database.dialect, PROMPT
                )
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, INTERMEDIATE_STEPS_KEY]

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
            sql_cmd = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()

            validate_sql(sql_cmd, self.database.dialect, self.sql_validation)

            if not self.use_query_checker:
                _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
                intermediate_steps.append(
                    sql_cmd
                )  # output: sql generation (no checker)
                intermediate_steps.append({"sql_cmd": sql_cmd})  # input: sql exec
                result = self.database.run(sql_cmd)
                intermediate_steps.append(str(result))  # output: sql exec
            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm_chain.llm, prompt=query_checker_prompt
                )
                query_checker_inputs = {
                    "query": sql_cmd,
                    "dialect": self.database.dialect,
                }
                checked_sql_command: str = query_checker_chain.predict(
                    callbacks=_run_manager.get_child(), **query_checker_inputs
                ).strip()
                intermediate_steps.append(
                    checked_sql_command
                )  # output: sql generation (checker)
                _run_manager.on_text(
                    checked_sql_command, color="green", verbose=self.verbose
                )
                intermediate_steps.append(
                    {"sql_cmd": checked_sql_command}
                )  # input: sql exec
                result = self.database.run(checked_sql_command)
                intermediate_steps.append(str(result))  # output: sql exec
                sql_cmd = checked_sql_command

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
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
        return "sql_database_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> SQLDatabaseChain:
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, **kwargs)


class SQLDatabaseSequentialChain(Chain):
    """Chain for querying SQL database that is a sequential chain.

    The chain is as follows:
    1. Based on the query, determine which tables to use.
    2. Based on those tables, call the normal SQL database chain.

    This is useful in cases where the number of tables in the database is large.
    """

    decider_chain: LLMChain
    sql_chain: SQLDatabaseChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database: SQLDatabase,
        query_prompt: BasePromptTemplate = PROMPT,
        decider_prompt: BasePromptTemplate = DECIDER_PROMPT,
        **kwargs: Any,
    ) -> SQLDatabaseSequentialChain:
        """Load the necessary chains."""
        sql_chain = SQLDatabaseChain.from_llm(
            llm, database, prompt=query_prompt, **kwargs
        )
        decider_chain = LLMChain(
            llm=llm, prompt=decider_prompt, output_key="table_names"
        )
        return cls(sql_chain=sql_chain, decider_chain=decider_chain, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, INTERMEDIATE_STEPS_KEY]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _table_names = self.sql_chain.database.get_usable_table_names()
        table_names = ", ".join(_table_names)
        llm_inputs = {
            "query": inputs[self.input_key],
            "table_names": table_names,
        }
        _lowercased_table_names = [name.lower() for name in _table_names]
        table_names_from_chain = self.decider_chain.predict_and_parse(**llm_inputs)
        table_names_to_use = [
            name
            for name in table_names_from_chain
            if name.lower() in _lowercased_table_names
        ]
        _run_manager.on_text("Table names to use:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(table_names_to_use), color="yellow", verbose=self.verbose
        )
        new_inputs = {
            self.sql_chain.input_key: inputs[self.input_key],
            "table_names_to_use": table_names_to_use,
        }
        return self.sql_chain(
            new_inputs, callbacks=_run_manager.get_child(), return_only_outputs=True
        )

    @property
    def _chain_type(self) -> str:
        return "sql_database_sequential_chain"
