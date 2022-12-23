from langchain.tracing.base import BaseTracer
from langchain.tracing.stdout import StdOutTracer


def get_tracer() -> BaseTracer:
    """Get the tracer."""
    return StdOutTracer()

