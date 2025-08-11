# Re-exporting internal utilities from LangGraph for internal use in LangChain.
# A different wrapper needs to be created for this purpose in LangChain.
from langgraph._internal._runnable import RunnableCallable

__all__ = [
    "RunnableCallable",
]
