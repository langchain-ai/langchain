# Runnable Interface

The Runnable interface is foundational for working with LangChain components, and it's implemented across many of them, such as [language models](/docs/concepts/chat_models), [output parsers](/docs/output_parsers), [retrievers](/docs/concepts/retrievers), [compiled langgraph graphs](
https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph) and more. Components that implement the Runnable interface can be combined using the [LangChain Expression Language (LCEL)](/docs/concepts/lcel) resulting a new Runnable that can be invoked, batched, streamed, and composed in a standard way.

This guide covers the main concepts and methods of the Runnable interface, which allows developers to interact with various LangChain components in a consistent and predictable manner.

:::note Related Resources
* The ["Runnable" Interface API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable) provides a detailed overview of the Runnable interface and its methods.
* A list of built-in `Runnables` can be found in the [LangChain Core API Reference](https://python.langchain.com/api_reference/core/runnables.html). Many of these Runnables are useful when composing custom "chains" in LangChain using the [LangChain Expression Language (LCEL)](/docs/concepts/lcel).
* The [LCEL cheatsheet](https://python.langchain.com/docs/how_to/lcel_cheatsheet/) shows common patterns that involve the Runnable interface and LCEL expressions.
:::

## Overview of Runnable Interface

The Runnable way defines a standard interface that allows a Runnable component to be:

* Invoked: A single input is transformed into an output.
* Batched: Multiple inputs are efficiently transformed into outputs.
* Streamed: Outputs are streamed as they are produced.
* Inspected: Schematic information about Runnable's input, output, and configuration can be accessed.
* Composed: Multiple Runnables can be composed to work together using [the LangChain Expression Language (LCEL)](/docs/concepts/lcel).

### Optimized Parallel Execution (Batch)

LangChain optimizes for parallel execution to minimize latency and increase throughput.

Batching options include `batch` and `batch_as_completed`:

* `batch`: Process multiple inputs in parallel, returning results in the same order as the inputs.
* `batch_as_completed`: Process multiple inputs in parallel, returning results as they complete. Results may arrive out of order, but each includes the input index for matching.

Using `batch` and `batch_as_completed` can significantly improve performance when needing to process multiple inputs concurrently.

The default `batch` and `batch_as_completed` implementations use a thread pool executor to run the `invoke` method in parallel. This allows for efficient parallel execution without the need for users to manage threads, and speeds up code that is I/O-bound (e.g., making API requests, reading files, etc.).

:::note
The async versions of `batch` and `batch_as_completed` are also available for asynchronous programming. These methods can be identified by the "a" prefix (e.g., `abatch`, `abatch_as_completed`). These rely on asyncio's [gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather) and [as_completed](https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed) functions to run the `ainvoke` method in parallel.
:::

:::tip
When working with LLMs and chat models, users may want to control the rate at which inputs are processed to avoid being rate-limited by the model provider.

Please see [RunnableConfig](/docs/concepts/runnables#runnableconfig) for more information on how to configure concurrency.
In addition, users have the option to specify a [rate limiter for chat models](/docs/concepts/chat_models#rate-limiting)
to have finer control over the rate of requests.
:::

### Asynchronous Support

Asynchronous programming is a paradigm that allows a program to perform multiple tasks concurrently without blocking the execution of other tasks, improving efficiency and responsiveness, particularly in I/O-bound operations

Runnables expose an asynchronous API, allowing them to be called using the `await` syntax in Python. This enables using the same code for prototypes and in production, providing great performance and the ability to handle many concurrent requests in the same server. Asynchronous methods can be identified by the "a" prefix (e.g., `ainvoke`, `abatch`, `astream`, `abatch_as_completed`).

:::note
Users are recommended to read the [asyncio documentation](https://docs.python.org/3/library/asyncio.html) to understand how to use the asynchronous programming
paradigm in Python.
:::

## Streaming

Streaming is critical in making applications based on LLMs feel responsive to end-users.

There are two general approaches to streaming the outputs from a Runnable:

* `stream` and `astream`: a default implementation of streaming that streams the output from the last step in the Runnable.
* async `astream_events` and the older async `astream_log`: these provide a way to stream both intermediate steps and final output from the chain.

There are few different ways of streaming outputs from a Runnable:

| Feature        | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| stream/astream | Streams output from a single input as it's produced.                                         |
| astream_events | Streams events as they happen asynchronously.                                                |
| astream_log    | Streams intermediate steps as they happen, in addition to the final response asynchronously. |

Please refer to the [Streaming Guide](/docs/concepts/streaming) for more details on streaming in LangChain.


## Inputs and Outputs

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

### Schemas

All Runnables expose input and output **schemas**. The schemas 

that can be used to inspect the inputs and outputs:

| Method                  | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `get_input_jsonschema`  | Gives the JSONSchema of the input schema for the Runnable.  |
| `get_output_jsonschema` | Gives the JSONSchema of the output schema for the Runnable. |
| `get_config_jsonschema` | Gives the JSONSchema of the config schema for the Runnable. |

This can be useful for validating inputs and outputs, as well as for generating documentation.

## RunnableConfig

Any of the methods that are used to execute the runnable (e.g., `invoke`, `batch`, `stream`, `astream_events`) accept a second argument called
`RunnableConfig` ([API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#runnableconfig)). This argument is a dictionary that contains configuration for the Runnable that will be used
at run time during the execution of the runnable.

A `RunnableConfig` can have any of the following properties defined:

| Attribute       | Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------|
| run_name        | Name used for the given Runnable (not inherited).                                          |
| run_id          | Unique identifier for the tracer run for this call.                                        |
| tags            | Tags for this call and any sub-calls.                                                      |
| metadata        | Metadata for this call and any sub-calls.                                                  |
| callbacks       | Callbacks for this call and any sub-calls.                                                 |
| max_concurrency | Maximum number of parallel calls to make (e.g., used by batch).                            |
| recursion_limit | Maximum number of times a call can recurse (e.g., used by Runnables that return Runnables) |
| configurable    | Runtime values for configurable attributes of the Runnable.                                |

## Customize Run Name

The `run_name`, `tags` and `metadata` are all attributes that can be used to customize 

You can specify a name for a given Runnable by setting the `run_name` attribute in the `RunnableConfig`. 

This information can be useful for debugging and tracing, as it allows you to easily identify the Runnable in logs and traces.

### Add Metadata and Tags to Traces

You can annotate a runs with arbitrary metadata and tags by providing them in the Config. This is useful for associating additional information with a trace, such as the environment in which it was executed, or the user who initiated it. 

## Customize Run ID

Useful only for advanced users


### Recursion Limit


### Max Concurrency


### Customize Metadata and Tags


### Customize Run ID

### Customize Callbacks

:::note Related

[Add Metadata and Tags to Traces](https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain#add-metadata-and-tags-to-traces)
:::



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
