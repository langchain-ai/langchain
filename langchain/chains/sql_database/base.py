"""Chain for interacting with SQL Database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.sql_database import SQLDatabase
from langchain.tools.sql_database.prompt import QUERY_CHECKER

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class SQLDatabaseChain(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain(llm=OpenAI(), database=db)
    """

    llm: BaseLanguageModel
    """LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    prompt: Optional[BasePromptTemplate] = None
    """Prompt to use to translate natural language to SQL."""
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

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompt or SQL_PROMPTS.get(self.database.dialect, PROMPT)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        self.callback_manager.on_text(input_text, verbose=self.verbose)
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
            intermediate_steps.append(llm_inputs)
            sql_cmd = llm_chain.predict(**llm_inputs)
            intermediate_steps.append(sql_cmd)
            if not self.use_query_checker:
                self.callback_manager.on_text(
                    sql_cmd, color="green", verbose=self.verbose
                )
                result = self.database.run(sql_cmd)
            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm, prompt=query_checker_prompt
                )
                query_checker_inputs = {
                    "query": sql_cmd,
                    "dialect": self.database.dialect,
                }
                intermediate_steps.append(query_checker_inputs)
                checked_sql_command: str = query_checker_chain.predict(
                    **query_checker_inputs
                )
                intermediate_steps.append(checked_sql_command)
                self.callback_manager.on_text(
                    checked_sql_command, color="green", verbose=self.verbose
                )
                result = self.database.run(checked_sql_command)
                sql_cmd = checked_sql_command

            intermediate_steps.append(result)
            self.callback_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            self.callback_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                self.callback_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
                llm_inputs["input"] = input_text
                final_result = llm_chain.predict(**llm_inputs)
                self.callback_manager.on_text(
                    final_result, color="green", verbose=self.verbose
                )
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


class SQLDatabaseSequentialChain(Chain):
    """Chain for querying SQL database that is a sequential chain.

    The chain is as follows:
    1. Based on the query, determine which tables to use.
    2. Based on those tables, call the normal SQL database chain.

    This is useful in cases where the number of tables in the database is large.
    """

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
        sql_chain = SQLDatabaseChain(
            llm=llm, database=database, prompt=query_prompt, **kwargs
        )
        decider_chain = LLMChain(
            llm=llm, prompt=decider_prompt, output_key="table_names"
        )
        return cls(sql_chain=sql_chain, decider_chain=decider_chain, **kwargs)

    decider_chain: LLMChain
    sql_chain: SQLDatabaseChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
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
        self.callback_manager.on_text(
            "Table names to use:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            str(table_names_to_use), color="yellow", verbose=self.verbose
        )
        new_inputs = {
            self.sql_chain.input_key: inputs[self.input_key],
            "table_names_to_use": table_names_to_use,
        }
        return self.sql_chain(new_inputs, return_only_outputs=True)

    @property
    def _chain_type(self) -> str:
        return "sql_database_sequential_chain"
