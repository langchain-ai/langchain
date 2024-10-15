# Chat Models

## Overview

Large Language Models (LLMs) are advanced machine learning models that excel in a wide range of language-related tasks such as text generation, translation, summarization, question answering, and more, without needing task-specific tuning for every scenario.

Modern LLMs are typically accessed through a chat model interface that takes [messages](/docs/concepts/messages) as input and returns [messages](/docs/concept/messages) as output.

In addition, some chat models offer additional capabilities:

* [Tool Calling](/docs/concepts#tool-calling): Many popular chat models offer a native [tool calling](/docs/concepts#tool-calling) API. Tool calling can be used to build rich applications that use AI to interact with external services, APIs, databases, to extract structured information from unstructured data and more.
* [Multimodality](/docs/concepts/multimodality): The ability to work with data other than text; for example, images, audio, and video.

## Why use LangChain?

LangChain provides a consistent interface for working with chat models from different providers while offering additional features for monitoring, debugging, and optimizing the performance of applications that use LLMs.

**Key Features**:

* Integrations with many chat model providers (e.g., Anthropic, OpenAI, Ollama, Cohere, Hugging Face, Groq, Microsoft Azure, Google Vertex, Amazon Bedrock). Please see [chat model integrations](/docs/integrations/chat_models/) for an up-to-date list of supported models.
* Use either LangChain's [messages](/docs/concepts/messages) format or OpenAI format.
* Standard [tool calling API](/docs/concepts#tool-calling): standard interface for binding tools to models, accessing tool call requests made by models, and sending tool results back to the model. 
* Provides support for [async programming](/docs/concepts/async), [efficient batching](/docs/concepts/runnables#batch), [a rich streaming API](/docs/concepts/streaming).
* Integration with [LangSmith](https://docs.smith.langchain.com) for monitoring and debugging production-grade applications based on LLMs.
* Standardized [token usage](/docs/concepts/messages).
* Additional features like: [rate limiting](#rate-limiting), [caching](#cache)

* Standardized model output metadata for easy integration with downstream applications; e.g., token usage, tool call requests etc.

## Naming Conventions

In the LangChain ecosystem, chat models are typically named with a convention that prefixes "Chat" to their class names (e.g., `ChatOllama`, `ChatAnthropic`, `ChatOpenAI`, etc.).

:::note
Do not confuse chat models with older LLMs, which are named without the "Chat" prefix (e.g., `Ollama`, `Anthropic`, `OpenAI`, etc.).
:::

## Runnable Interface

LangChain chat models implement the [Runnable Interface](/docs/concepts/runnables) which provides a [standard streaming interface](/docs/concepts/streaming), support for [async programming](/docs/concepts/async), optimized [batching](/docs/concepts/runnables#batch), and more.

## Multimodality

Large Language Models (LLMs) are not limited to processing text. They can also be used to process other types of data, such as images, audio, and video. This is known as [multimodality](/docs/concepts/multimodality).

Currently, only some LLMs support multimodal inputs, and almost none support multimodal outputs. Please consult the specific model documentation for details.

LLMs are best thought of as models that operate on sequences of tokens to predict the next token in a sequence. Tokens are abstract representations of input data that can take a variety of forms, such as text, code, images, audio, video, and more.

* Learn more about [MultiModality](/docs/concepts/multimodality).

## Tokenization

Despite their name, LLMs are not limited to processing natural language text. The underlying technology, based on the transformer architecture, operates on sequences of tokens. These tokens are abstract representations of input data
which can take a variety of forms, such as text, code, images, audio, and more.

* Read more about [Tokenization](/docs/concepts/tokenization).


## LangChain?

LangChain does not host any Chat Models, rather we rely on third party integrations.

## Standardization

### Standard Parameters

We have some standardized parameters when constructing ChatModels:

| Parameter     | Description                                                                     |
|---------------|---------------------------------------------------------------------------------|
| `model`       | the name of the model                                                           |
| `temperature` | the sampling temperature; higher values result in more randomness in the output |
| `timeout`     | request timeout                                                                 |
| `max_tokens`  | max tokens to generate                                                          |
| `stop`        | default stop sequences                                                          |
| `max_retries` | max number of times to retry requests                                           |
| `api_key`     | API key for the model provider                                                  |
| `base_url`    | endpoint to send requests to                                                    |

Some important things to note:
- standard parameters only apply to model providers that expose parameters with the intended functionality. For example, some providers do not expose a configuration for maximum output tokens, so max_tokens can't be supported on these.
- standard params are currently only enforced on integrations that have their own integration packages (e.g. `langchain-openai`, `langchain-anthropic`, etc.), they're not enforced on models in ``langchain-community``.

ChatModels also accept other parameters that are specific to that integration. To find all the parameters supported by a ChatModel head to the API reference for that model.

:::important
Some chat models have been fine-tuned for **tool calling** and provide a dedicated API for it.
Generally, such models are better at tool calling than non-fine-tuned models, and are recommended for use cases that require tool calling.
Please see the [tool calling section](/docs/concepts/#functiontool-calling) for more information.
:::

For specifics on how to use chat models, see the [relevant how-to guides here](/docs/how_to/#chat-models).

## Messages

Chat models take a sequence of messages as input and return messages as output. Messages are structured data that contain the text of the message, the role of the speaker, and any other relevant metadata.

Although the underlying models are messages in, message out, the LangChain wrappers also allow these models to take a string as input. This means you can easily use chat models in place of LLMs.

When a string is passed in as input, it is converted to a `HumanMessage` and then passed to the underlying model.


## Cache

### Should chat model results be cached?

* Cache is available, but should be exercised with caution.
* Cache hits are unlikely below the first or second level of conversation if using exact matches.
* Would need to use a semantic cache to get more hits, but even then conceptually it is not a good idea to cache too much

## Tokenization

Tokenization refers to the process of breaking down text into smaller units called tokens. These tokens can be words, subwords, or even characters, depending on the method used. Tokenization is essential because most NLP models work with structured inputs, and tokenization allows for converting raw text into a form that models can process effectively.

Please see the [tokenization section](/docs/concepts/#text-splitting) for more information.

## Retries

## Rate-limiting

Many chat model providers impose a limit on the number of requests that can be made in a given time period.

If you hit a rate limit, you will typically receive a rate limit error response from the provider, and will need to wait before making more requests.

There are a few different strategies for dealing with rate limits:

1. **Backoff**: If you receive a rate limit error, you can wait a certain amount of time before retrying the request. The amount of time to wait can be increased with each subsequent rate limit error. Some of the chat models in LangChain have built-in backoff and retry mechanisms.
2. Using different model providers

There are two strategies for dealing with rate limits:


:::note
:::



## In LangChain

These are traditionally older models (newer models generally are [Chat Models](/docs/concepts/#chat-models), see above).

Although the underlying models are string in, string out, the LangChain wrappers also allow these models to take messages as input.
This gives them the same interface as [Chat Models](/docs/concepts/#chat-models).
When messages are passed in as input, they will be formatted into a string under the hood before being passed to the underlying model.

LangChain does not host any LLMs, rather we rely on third party integrations.

For specifics on how to use LLMs, see the [how-to guides](/docs/how_to/#llms).
