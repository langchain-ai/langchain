from functools import partial
from typing import Any, Optional
from typing_extensions import TypeAlias
from langchain.callbacks.manager import CallbackManagerForToolRun

from langchain.prompts.prompt import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable.base import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool

DF: TypeAlias = Any


def evaluate_sql_on_dfs(sql: str, **dfs: DF) -> DF:
    """Evaluate a SQL query on a pandas dataframe."""
    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required to evaluate SQL queries on pandas dataframes."
        )

    locals().update(dfs)
    conn = duckdb.connect()
    return conn.execute(sql).fetchall()


def get_pandas_eval_chain(model: BaseLanguageModel, dfs: dict[str, DF]) -> Runnable:
    prompt = PromptTemplate.from_template(
        """You are an expert data scientist, tasked with converting python code manipulating pandas dataframes into SQL queries.

You should write a SQL query that will return the same result as the python code below/There are SQL tables with the same name as any Pandas dataframe in the code.

You are given the following python code:

{input}

SQL query:"""
    )

    return prompt | model | StrOutputParser() | partial(evaluate_sql_on_dfs, **dfs)


class PandasEvalTool(BaseTool):
    name: str = "pandas_eval"

    description: str = "Evaluate pandas code against one or more dataframes."

    dfs: dict[str, DF]

    model: BaseLanguageModel

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        chain = get_pandas_eval_chain(self.model, self.dfs)
        return chain.invoke({"input": query}, {"callbacks": run_manager.get_child()})
