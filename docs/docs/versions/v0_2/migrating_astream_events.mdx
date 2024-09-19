---
sidebar_position: 2
sidebar_label: astream_events v2
---

# Migrating to astream_events(..., version="v2")

We've added a `v2` of the astream_events API with the release of `0.2.x`. You can see this [PR](https://github.com/langchain-ai/langchain/pull/21638) for more details.

The `v2` version is a re-write of the `v1` version, and should be more efficient, with more consistent output for the events. The `v1` version of the API will be deprecated in favor of the `v2` version and will be removed in  `0.4.0`.

Below is a list of changes between the `v1` and `v2` versions of the API.


### output for `on_chat_model_end`

In `v1`, the outputs associated with `on_chat_model_end` changed depending on whether the
chat model was run as a root level runnable or as part of a chain.

As a root level runnable the output was:

```python
"data": {"output": AIMessageChunk(content="hello world!", id='some id')}
```

As part of a chain the output was:

```
            "data": {
                "output": {
                    "generations": [
                        [
                            {
                                "generation_info": None,
                                "message": AIMessageChunk(
                                    content="hello world!", id=AnyStr()
                                ),
                                "text": "hello world!",
                                "type": "ChatGenerationChunk",
                            }
                        ]
                    ],
                    "llm_output": None,
                }
            },
```


As of `v2`, the output will always be the simpler representation:

```python
"data": {"output": AIMessageChunk(content="hello world!", id='some id')}
```

:::note
Non chat models (i.e., regular LLMs) are will be consistently associated with the more verbose format for now.
:::

### output for `on_retriever_end`

`on_retriever_end` output will always return a list of `Documents`.

Before:
```python
{
    "data": {
        "output": [
            Document(...),
            Document(...),
            ...
        ]
    }
}
```

### Removed `on_retriever_stream`

The `on_retriever_stream` event was an artifact of the implementation and has been removed.

Full information associated with the event is already available in the `on_retriever_end` event.

Please use `on_retriever_end` instead.

### Removed `on_tool_stream`

The `on_tool_stream` event was an artifact of the implementation and has been removed.

Full information associated with the event is already available in the `on_tool_end` event.

Please use `on_tool_end` instead.

### Propagating Names

Names of runnables have been updated to be more consistent.

```python
model = GenericFakeChatModel(messages=infinite_cycle).configurable_fields(
    messages=ConfigurableField(
        id="messages",
        name="Messages",
        description="Messages return by the LLM",
    )
)
```

In `v1`, the event name was `RunnableConfigurableFields`.

In `v2`, the event name is `GenericFakeChatModel`.

If you're filtering by event names, check if you need to update your filters.

### RunnableRetry

Usage of [RunnableRetry](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.retry.RunnableRetry.html)
within an LCEL chain being streamed generated an incorrect `on_chain_end` event in `v1` corresponding
to the failed runnable invocation that was being retried. This event has been removed in `v2`.

No action is required for this change.
