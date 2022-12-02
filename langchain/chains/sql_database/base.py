"""Chain for interacting with SQL Database."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import PROMPT
from langchain.input import ChainedInput
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        chained_input = ChainedInput(
            f"{inputs[self.input_key]} \nSQLQuery:", verbose=self.verbose
        )
        llm_inputs = {
            "input": chained_input.input,
            "dialect": self.database.dialect,
            "table_info": self.database.table_info,
            "stop": ["\nSQLResult:"],
        }
        sql_cmd = llm_chain.predict(**llm_inputs)
        chained_input.add(sql_cmd, color="green")
        result = self.database.run(sql_cmd)
        chained_input.add("\nSQLResult: ")
        chained_input.add(result, color="yellow")
        chained_input.add("\nAnswer:")
        llm_inputs["input"] = chained_input.input
        final_result = llm_chain.predict(**llm_inputs)
        chained_input.add(final_result, color="green")
        return {self.output_key: final_result}
