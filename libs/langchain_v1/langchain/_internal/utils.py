# Re-exporting internal utilities from LangGraph for internal use in LangChain.
# TODO: We need to revisit the solution. Perhaps we expose a simple wrapper in langgraph
# create_node(sync, async) that will return a new node or something like that
from langgraph._internal._runnable import RunnableCallable

__all__ = [
    "RunnableCallable",
]
