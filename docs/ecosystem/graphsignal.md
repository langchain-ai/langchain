# Graphsignal

This page covers how to use the Graphsignal to trace and monitor LangChain.

## Installation and Setup

- Install the Python library with `pip install graphsignal`
- Create free Graphsignal account [here](https://graphsignal.com)
- Get an API key and set it as an environment variable (`GRAPHSIGNAL_API_KEY`)

## Tracing and Monitoring

Graphsignal automatically instruments and starts tracing and monitoring chains. Traces, metrics and errors are then available in your [Graphsignal dashboard](https://app.graphsignal.com/). No prompts or other sensitive data are sent to Graphsignal cloud, only statistics and metadata.

Initialize the tracer by providing a deployment name:

```python
import graphsignal

graphsignal.configure(deployment='my-langchain-app-prod')
```

In order to trace full runs and see a breakdown by chains and tools, you can wrap the calling routine or use a decorator:

```python
with graphsignal.start_trace('my-chain'):
    chain.run("some initial text")
```

Optionally, enable profiling to record function-level statistics for each trace.

```python
with graphsignal.start_trace(
        'my-chain', options=graphsignal.TraceOptions(enable_profiling=True)):
    chain.run("some initial text")
```

See the [Quick Start](https://graphsignal.com/docs/guides/quick-start/) guide for complete setup instructions.
