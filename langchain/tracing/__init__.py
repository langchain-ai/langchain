import os

from langchain.tracing.base import BaseTracer
from langchain.tracing.nested_json import NestedJsonTracer
from langchain.tracing.noop import NoOpTracer
from langchain.tracing.stdout import StdOutTracer


def get_tracer() -> BaseTracer:
    """Get the tracer."""

    if "LANGCHAIN_TRACER" in os.environ:
        tracer = os.environ["LANGCHAIN_TRACER"]
        if tracer == "stdout":
            return StdOutTracer()
        elif tracer == "nested_json":
            return NestedJsonTracer()
        else:
            raise ValueError(f"Unknown tracer {tracer}")

    return NoOpTracer()
