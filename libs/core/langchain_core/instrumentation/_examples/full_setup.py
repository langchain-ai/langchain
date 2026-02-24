"""Full setup: Prometheus + OpenTelemetry + LangSmith running simultaneously.

This is a runnable example showing how all three observability backends
work together through CompositeProvider.

Run:
    pip install prometheus-client opentelemetry-api opentelemetry-sdk langchain-openai
    export OPENAI_API_KEY=sk-...
    export LANGCHAIN_TRACING_V2=true        # enables LangSmith
    export LANGCHAIN_API_KEY=ls-...
    python -m langchain_core.instrumentation._examples.full_setup

After running:
    - Prometheus metrics → http://localhost:9090/metrics
    - OTel traces → console (or Jaeger if you configure an OTLP exporter)
    - LangSmith traces → https://smith.langchain.com
"""

from __future__ import annotations


def main() -> None:
    # ──────────────────────────────────────────────────────────
    # 1. Setup OpenTelemetry tracer → exports to console
    # ──────────────────────────────────────────────────────────
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    otel_provider = TracerProvider()
    otel_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(otel_provider)
    tracer = trace.get_tracer("langchain")

    # ──────────────────────────────────────────────────────────
    # 2. Setup Prometheus metrics → http://localhost:9090/metrics
    # ──────────────────────────────────────────────────────────
    from langchain_core.instrumentation._examples.prometheus_provider import (
        PrometheusMetricsProvider,
    )

    prometheus = PrometheusMetricsProvider(port=9090)
    print("Prometheus metrics → http://localhost:9090/metrics")

    # ──────────────────────────────────────────────────────────
    # 3. LangSmith stays enabled via LANGCHAIN_TRACING_V2=true
    #    (no code changes needed — callbacks work as before)
    #
    #    Optionally, wrap existing callbacks in a bridge:
    # ──────────────────────────────────────────────────────────
    # from langchain_core.callbacks import CallbackManager
    # from langchain_core.tracers import LangChainTracer
    # langsmith_bridge = CallbackBridgeProvider(
    #     CallbackManager(handlers=[LangChainTracer()])
    # )
    # → then add langsmith_bridge to CompositeProvider below

    # ──────────────────────────────────────────────────────────
    # 4. Combine into CompositeProvider
    # ──────────────────────────────────────────────────────────
    from langchain_core.instrumentation import (
        CompositeProvider,
        set_instrumentation_provider,
    )
    from langchain_core.instrumentation._examples.otel_provider import OTelProvider

    otel = OTelProvider(tracer)

    set_instrumentation_provider(
        CompositeProvider([
            otel,           # OTel spans → console / Jaeger / Tempo
            prometheus,     # Prometheus counters + histograms → :9090
        ])
    )
    # LangSmith works independently via callbacks (LANGCHAIN_TRACING_V2=true)

    # ──────────────────────────────────────────────────────────
    # 5. Use LangChain normally — everything is instrumented
    # ──────────────────────────────────────────────────────────
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("\n--- Calling model.invoke() ---\n")
    response = model.invoke("What is the capital of France? One word.")
    print(f"Response: {response.content}")

    print("\n--- Calling model.invoke() again ---\n")
    response = model.invoke("What is 2 + 2? One number.")
    print(f"Response: {response.content}")

    # ──────────────────────────────────────────────────────────
    # 6. Check what we collected
    # ──────────────────────────────────────────────────────────
    print("\n--- Prometheus metrics available at http://localhost:9090/metrics ---")
    print("Press Ctrl+C to exit")
    print("\nExample metrics you'll see:")
    print("  langchain_tokens_total{model='gpt-4o-mini',direction='input'}")
    print("  langchain_tokens_total{model='gpt-4o-mini',direction='output'}")
    print("  langchain_llm_duration_seconds_bucket{model='gpt-4o-mini',le='1.0'}")
    print("  langchain_errors_total{kind='llm'}")

    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
