from langchain.tracing.base import BaseTracer
from langchain.tracing.nested_json import NestedJsonTracer
from langchain.tracing.stdout import StdOutTracer


def get_tracer() -> BaseTracer:
    """Get the tracer."""

    return NestedJsonTracer()
