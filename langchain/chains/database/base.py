from langchain.chains.base import Chain
from typing import List, Dict
from langchain.llms.base import LLM
from langchain.chains.database.prompt import PROMPT
from langchain.chains.llm import LLMChain
from langchain.databases.base import Database
from pydantic import BaseModel, Extra


class DatabaseChain(Chain, BaseModel):


    llm: LLM
    """LLM wrapper to use."""
    database: Database
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
        llm_inputs = {
            "input": inputs[self.input_key],
            "dialect": self.database.dialect,
            "table_info": self.database.table_info
        }
        sql_cmd = llm_chain.predict(**llm_inputs)
        print(sql_cmd)
        result = self.database.run(sql_cmd)
        return {self.output_key: result}

    def query(self, query: str) -> str:
        """Run natural language query against database."""
        return self({self.input_key: query})[self.output_key]

