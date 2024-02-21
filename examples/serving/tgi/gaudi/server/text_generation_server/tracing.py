import grpc

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc._aio_server import (
    OpenTelemetryAioServerInterceptor,
)
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)


class UDSOpenTelemetryAioServerInterceptor(OpenTelemetryAioServerInterceptor):
    def __init__(self):
        super().__init__(trace.get_tracer(__name__))

    def _start_span(self, handler_call_details, context, set_status_on_exception=False):
        """
        Rewrite _start_span method to support Unix Domain Socket gRPC contexts
        """

        # standard attributes
        attributes = {
            SpanAttributes.RPC_SYSTEM: "grpc",
            SpanAttributes.RPC_GRPC_STATUS_CODE: grpc.StatusCode.OK.value[0],
        }

        # if we have details about the call, split into service and method
        if handler_call_details.method:
            service, method = handler_call_details.method.lstrip("/").split("/", 1)
            attributes.update(
                {
                    SpanAttributes.RPC_METHOD: method,
                    SpanAttributes.RPC_SERVICE: service,
                }
            )

        # add some attributes from the metadata
        metadata = dict(context.invocation_metadata())
        if "user-agent" in metadata:
            attributes["rpc.user_agent"] = metadata["user-agent"]

        # We use gRPC over a UNIX socket
        attributes.update({SpanAttributes.NET_TRANSPORT: "unix"})

        return self._tracer.start_as_current_span(
            name=handler_call_details.method,
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
            set_status_on_exception=set_status_on_exception,
        )


def setup_tracing(shard: int, otlp_endpoint: str):
    resource = Resource.create(
        attributes={"service.name": f"text-generation-inference.server-{shard}"}
    )
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    span_processor = BatchSpanProcessor(span_exporter)

    trace.set_tracer_provider(TracerProvider(resource=resource))
    trace.get_tracer_provider().add_span_processor(span_processor)
