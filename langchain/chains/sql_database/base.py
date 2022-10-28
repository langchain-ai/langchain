"""Chain for interacting with SQL Database."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import PROMPT
from langchain.llms.base import LLM
from langchain.sql_database import SQLDatabase


class SQLDatabaseChain(Chain, BaseModel):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SelfAskWithSearchChain(llm=OpenAI(), database=db)
    """

    llm: LLM
    """LLM wrapper to use."""
    database: SQLDatabase
    """SQL Database to connect to."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

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
        return [self.output_key]

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        _input = inputs[self.input_key] + "\nSQLQuery:"
        llm_inputs = {
            "input": _input,
            "dialect": self.database.dialect,
            "table_info": self.database.table_info,
            "stop": ["\nSQLResult:"],
        }
        sql_cmd = llm_chain.predict(**llm_inputs)
        print(sql_cmd)
        result = self.database.run(sql_cmd)
        print(result)
        _input += f"\nSQLResult: {result}\nAnswer:"
        llm_inputs["input"] = _input
        final_result = llm_chain.predict(**llm_inputs)
        return {self.output_key: final_result}

    def query(self, query: str) -> str:
        """Run natural language query against a SQL database.

        Args:
            query: natural language query to run against the SQL database

        Returns:
            The final answer as derived from the SQL database.

        Example:
            .. code-block:: python

                answer = db_chain.query("How many customers are there?")
        """
        return self({self.input_key: query})[self.output_key]
