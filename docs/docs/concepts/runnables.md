# Runnable Interface

The Runnable interface is foundational for working with LangChain components, and it's implemented across many of them, such as [language models](/docs/concepts/chat_models), [output parsers](/docs/output_parsers), [retrievers](/docs/concepts/retrievers), [compiled LangGraph graphs](
https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph) and more. Components that implement the Runnable interface can be combined using the [LangChain Expression Language (LCEL)](/docs/concepts/lcel) resulting a new Runnable that can be invoked, batched, streamed, and composed in a standard way.

This guide covers the main concepts and methods of the Runnable interface, which allows developers to interact with various LangChain components in a consistent and predictable manner.

:::note Related Resources
* The ["Runnable" Interface API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable) provides a detailed overview of the Runnable interface and its methods.
* A list of built-in `Runnables` can be found in the [LangChain Core API Reference](https://python.langchain.com/api_reference/core/runnables.html). Many of these Runnables are useful when composing custom "chains" in LangChain using the [LangChain Expression Language (LCEL)](/docs/concepts/lcel).
* The [LCEL cheatsheet](https://python.langchain.com/docs/how_to/lcel_cheatsheet/) shows common patterns that involve the Runnable interface and LCEL expressions.
:::

## Overview of Runnable Interface

The Runnable way defines a standard interface that allows a Runnable component to be:

* [Invoked](/docs/how_to/lcel_cheatsheet/#invoke-a-runnable): A single input is transformed into an output.
* [Batched](/docs/how_to/lcel_cheatsheet/#batch-a-runnable/): Multiple inputs are efficiently transformed into outputs.
* [Streamed](/docs/how_to/lcel_cheatsheet/#stream-a-runnable): Outputs are streamed as they are produced.
* Inspected: Schematic information about Runnable's input, output, and configuration can be accessed.
* Composed: Multiple Runnables can be composed to work together using [the LangChain Expression Language (LCEL)](/docs/concepts/lcel).

Please review the [LCEL Cheatsheet](/docs/how_to/lcel_cheatsheet) for some common patterns that involve the Runnable interface and LCEL expressions.

### Optimized Parallel Execution (Batch)

LangChain Runnables offer a built-in `batch` (and `batch_as_completed`) API that allow you to process multiple inputs in parallel.

Using these methods can significantly improve performance when needing to process multiple independent inputs, as the
processing can be done in parallel instead of sequentially.

The two batching options are:

* `batch`: Process multiple inputs in parallel, returning results in the same order as the inputs.
* `batch_as_completed`: Process multiple inputs in parallel, returning results as they complete. Results may arrive out of order, but each includes the input index for matching.

The default implementation of `batch` and `batch_as_completed` use a thread pool executor to run the `invoke` method in parallel. This allows for efficient parallel execution without the need for users to manage threads, and speeds up code that is I/O-bound (e.g., making API requests, reading files, etc.). It will not be as effective for CPU-bound operations, as the GIL (Global Interpreter Lock) in Python will prevent true parallel execution.

Some Runnables may provide their own implementations of `batch` and `batch_as_completed` that are optimized for their specific use case (e.g.,
rely on a `batch` API provided by a model provider).

:::note
The async versions of `abatch` and `abatch_as_completed` these rely on asyncio's [gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather) and [as_completed](https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed) functions to run the `ainvoke` method in parallel.
:::

:::tip
When processing a large number of inputs using `batch` or `batch_as_completed`, users may want to control the maximum number of parallel calls. This can be done by setting the `max_concurrency` attribute in the `RunnableConfig` dictionary. See the [RunnableConfig](/docs/concepts/runnables#runnableconfig) for more information.

Chat Models also have a built-in [rate limiter](/docs/concepts/chat_models#rate-limiting) that can be used to control the rate at which requests are made.
:::

### Asynchronous Support

Asynchronous programming is a paradigm that allows a program to perform multiple tasks concurrently without blocking the execution of other tasks, improving efficiency and responsiveness, particularly in I/O-bound operations

Runnables expose an asynchronous API, allowing them to be called using the `await` syntax in Python. This enables using the same code for prototypes as would be used in production, providing great performance and the ability to handle many concurrent requests on the same server. 

Asynchronous methods can be identified by the "a" prefix (e.g., `ainvoke`, `abatch`, `astream`, `abatch_as_completed`).

:::note
Users are recommended to read the [asyncio documentation](https://docs.python.org/3/library/asyncio.html) to understand how to use the asynchronous programming paradigm in Python.
:::

## Streaming APIs

Streaming is critical in making applications based on LLMs feel responsive to end-users.

Runnables expose the following three streaming APIs:

1. The `stream` and `astream`: a default implementation of streaming that streams the final output
2. The async `astream_events`: a more advanced streaming API that allows streaming intermediate steps and final output  
3. The **legacy** async `astream_log`: a legacy streaming API that streams intermediate steps and final output

Please refer to the [Streaming Conceptual Guide](/docs/concepts/streaming) for more details on how to stream in LangChain.

## Input and Output Types

A Runnable is characterized by an input and output type. These input and output types can be any Python object, and are defined by the Runnable itself.

Runnable methods that result in the execution of the Runnable (e.g., `invoke`, `batch`, `stream`, `astream_events`) work with these input and output types.

* invoke: Accepts an input and returns an output.
* batch: Accepts a list of inputs and returns a list of outputs.
* stream: Accepts an input and returns a generator that yields outputs.

The **input type** and **output type** varies by component:

| Component    | Input Type                                       | Output Type           |
|--------------|--------------------------------------------------|-----------------------|
| Prompt       | dictionary                                       | PromptValue           |
| ChatModel    | a string, list of chat messages or a PromptValue | ChatMessage           |
| LLM          | a string, list of chat messages or a PromptValue | String                |
| OutputParser | the output of an LLM or ChatModel                | Depends on the parser |
| Retriever    | a string                                         | List of Documents     |
| Tool         | a string or dictionary, depending on the tool    | Depends on the tool   |

Please refer to the individual component documentation for more information on the input and output types and how to use them.

### JSON Schemas

In some situations, you may want to determine what input and output types the Runnable expects and produces. 

You can use the `get_input_jsonschema` and `get_output_jsonschema` methods to get the [JSON Schema](https://json-schema.org/) of the input and output types of a Runnable.

This can be useful for validating inputs and outputs, as well as for generating [OpenAPI documentation](https://www.openapis.org/).

Runnables have an API that allows you to get the [JSON Schema](https://json-schema.org/) of the input and output of a Runnable. This can be useful for validating inputs and outputs, as well as for generating [OpenAPI documentation](https://www.openapis.org/).

In addition to inputs and outputs some `Runnables` can be set up to be configurable at run time with additional parameters.

Please see the [Configurable Runnables](#configurable-runnables) section for more information. The configuration schema can be accessed using the `get_config_jsonschema` method of the Runnable.

| Method                  | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `get_input_jsonschema`  | Gives the JSONSchema of the input schema for the Runnable.  |
| `get_output_jsonschema` | Gives the JSONSchema of the output schema for the Runnable. |
| `get_config_jsonschema` | Gives the JSONSchema of the config schema for the Runnable. |


#### with_types

LangChain will automatically try to infer the input and output types of a Runnable based on available information.

In some situations, especially when building more complex Runnables using [LCEL](/docs/concepts/lcel) composition, the inferred input and / or output types will be incorrect, and need to be overridden.

To address this, the `with_types` method allows you to explicitly set the input and output types of a Runnable.


## RunnableConfig

Any of the methods that are used to execute the runnable (e.g., `invoke`, `batch`, `stream`, `astream_events`) accept a second argument called
`RunnableConfig` ([API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#runnableconfig)). This argument is a dictionary that contains configuration for the Runnable that will be used
at run time during the execution of the runnable.

A `RunnableConfig` can have any of the following properties defined:

| Attribute       | Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------|
| run_name        | Name used for the given Runnable (not inherited).                                          |
| run_id          | Unique identifier for this call. sub-calls will get their own unique run ids.              |
| tags            | Tags for this call and any sub-calls.                                                      |
| metadata        | Metadata for this call and any sub-calls.                                                  |
| callbacks       | Callbacks for this call and any sub-calls.                                                 |
| max_concurrency | Maximum number of parallel calls to make (e.g., used by batch).                            |
| recursion_limit | Maximum number of times a call can recurse (e.g., used by Runnables that return Runnables) |
| configurable    | Runtime values for configurable attributes of the Runnable.                                |

### Propagation of RunnableConfig

A `Runnable` can be assembled from other `Runnables` via the LangChain Expression Language (LCEL); e.g.,

```python
chain = prompt | chat_model | output_parser
```

In addition, runnables can invoke other runnables as part of imperative code; e.g., 

```python
def foo(input):
    return bar_runnable.invoke(input)

foo_runnable = RunnableLambda(foo)
```

# WIP BELOW THIS LINE

`RunableConfig`


In this case, the `RunnableConfig` is propagated to all sub-calls made by the `Runnable`. This allows you to set configuration values for the entire call chain, and have them inherited by all sub-calls.

When a `Runnable` is assembled from other `Runnables`, the `RunnableConfig` is propagated to all sub-calls made by the `Runnable`. This allows you to set configuration values for the entire call chain, and have them inherited by all sub-calls.



The `RunnableConfig` is propagated to all sub-calls made by the Runnable. 

This allows you to set configuration values for the entire call chain, and have them inherited by all sub-calls.


### Custom Metadata and Tags

You can provide custom metadata and tags to a Runnable by setting the `metadata` and `tags` attributes in the `RunnableConfig`.


The `RunnableConfig` can be used to customize the behavior of a Runnable in a number of ways.

The `run_name`, `tags` and `metadata` are all attributes that can be used to customize 

You can specify a name for a given Runnable by setting the `run_name` attribute in the `RunnableConfig`. 

This information can be useful for debugging and tracing, as it allows you to easily identify the Runnable in logs and traces.

### Add Metadata and Tags to Traces

You can annotate a runs with arbitrary metadata and tags by providing them in the Config. This is useful for associating additional information with a trace, such as the environment in which it was executed, or the user who initiated it.

:::note Tracing with LangSmith
[How to customize attributes of traces](https://docs.smith.langchain.com/old/tracing/faq/customizing_trace_attributes)
:::

## Customize Run ID

In some situations, you may need to set the `run_id` of the Runnable, so you can
do something with that ID later on in the application. 

```python

import uuid

run_id = str(uuid.uuid4())
runnable.invoke(inputs, config={'run_id': run_id})
```

### Recursion Limit


### Max Concurrency


### Customize Metadata and Tags


### Customize Run ID

### Customize Callbacks


## Debugging and Tracing

The standard interface includes a number of methods for debugging and tracing, including:

- **Callbacks**: You can pass existing or custom callbacks to any given chain to debug and trace the chain.
- **Debugging**: You can set the global debug flag to True to enable debug output for all chains.

You can set the global debug flag to True to enable debug output for all chains:

      .. code-block:: python

          from langchain_core.globals import set_debug
          set_debug(True)

Alternatively, you can pass existing or custom callbacks to any given chain:

      .. code-block:: python

          from langchain_core.tracers import ConsoleCallbackHandler

          chain.invoke(
              ...,
              config={'callbacks': [ConsoleCallbackHandler()]}
          )


## Custom Runnables

Users should create custom Runnables by creating either a `RunnableLambda` or a `RunnableGenerator`.

Users create a custom Runnable by

For more complex workflows, users can create custom Runnables by extending the interface. There are two main types of custom Runnables:

* `RunnableLambda`: For simple transformations where streaming is not required.
* `RunnableGenerator`: For more complex transformations when streaming is needed.


## RunnableLambda


## RunnableGenerator


## Configurable Runnables
