from typing import Any


def __getattr__(name: str) -> Any:
    """Get attr name."""
    if name == "create_pandas_dataframe_agent":
        msg = (
            "This agent has been moved to langchain experiment. "
            "This agent relies on python REPL tool under the hood, so to use it "
            "safely please sandbox the python REPL. "
            "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
            "and https://github.com/langchain-ai/langchain/discussions/11680"
            "To keep using this code as is, install langchain experimental and "
            "update your import statement from:\n"
            f"`langchain_classic.agents.agent_toolkits.pandas.{name}` to "
            f"`langchain_experimental.agents.agent_toolkits.{name}`."
        )
        raise ImportError(msg)
    msg = f"{name} does not exist"
    raise AttributeError(msg)
