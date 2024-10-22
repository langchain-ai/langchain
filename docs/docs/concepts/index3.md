# Conceptual Guide

## Overview

In this guide, you'll find explanations of key concepts, providing a deeper understanding of core principles. We recommend that you go through at least one of the [Tutorials](/docs/tutorials) before diving into the conceptual guide to help you understand the context and practical applications of the concepts discussed here. The conceptual guide will not cover step-by-step instructions or specific implementation details — those are found in the [How-To Guides](/docs/how_to/) and [Tutorials](/docs/tutorials) sections. For detailed reference material, visit the [API Reference](https://python.langchain.com/api_reference/).

## Concepts

- **[Why LangChain?](/docs/concepts/why_langchain)**: Why LangChain is the best choice for building AI applications.

- **[Architecture](/docs/concepts/architecture)**: Overview of how packages are organized in the LangChain ecosystem. **Terms**: [langchain](/docs/concepts/architecture#langchain), [langchain-community](/docs/concepts/architecture#langchain-community), [langchain-core](/docs/concepts/architecture#langchain-core), [langserve](/docs/concepts/architecture#langserve), [langgraph](/docs/concepts/architecture#langgraph), [partner-packages](/docs/concepts/architecture#partner-packages).

- **[Chat Models](/docs/concepts/chat_models)**: Modern LLMs exposed via a chat interface that process sequences of messages as input and output a message. **Terms**: [key methods](/docs/concepts/chat_models#key-methods), [standard parameters](/docs/concepts/chat_models#standard-parameters), [context-window](/docs/concepts/chat_models#context-window), [with-structured-output](/docs/concepts/chat_models#with-structured-output), [bind-tools](/docs/concepts/chat_models#bind-tools), [multimodality](/docs/concepts/chat_models#multimodality), [rate-limiting](/docs/concepts/chat_models#rate-limiting), [caching](/docs/concepts/chat_models#caching).

- **[Messages](/docs/concepts/messages)**: Messages are the units of communication in modern LLMs, representing input and output, and any additional context or metadata associated with the conversation. **Terms**: [role](/docs/concepts/messages#role), [content](/docs/concepts/messages#content), [HumanMessage](/docs/concepts/messages#humanmessage), [AIMessage](/docs/concepts/messages#aimessage), [ToolMessage](/docs/concepts/messages#toolmessage), [AIMessageChunk](/docs/concepts/messages#aimessagechunk), [RemoveMessage](/docs/concepts/messages#remove-message), [OpenAI format](/docs/concepts/messages#openai-format).

- **[Chat History](/docs/concepts/chat_history)**: Chat history is a record of the conversation between the user and the chat model, used to maintain context and state. **Terms**: [conversation-patterns](/docs/concepts/chat_history#conversation-patterns), [managing-chat-history](/docs/concepts/chat_history#managing-chat-history).

- **[Tools](/docs/concepts/tools)**: The tool abstraction in LangChain associates a Python function with a schema defining the function's name, description, and input. Tools can be passed to chat models that support [tool calling](/docs/concepts/tool_calling). **Terms**: [@tool](/docs/concepts/tools#@tool), [BaseTool](/docs/concepts/tools#basetool), [Tool-artifacts](/docs/concepts/tools#tool-artifacts), [InjectedToolArg](/docs/concepts/tools#injectedtoolarg), [RunnableConfig](/docs/concepts/tools#runnableconfig), [InjectedState](/docs/concepts/tools#injectedstate), [InjectedStore](/docs/concepts/tools#injectedstore).

- **[Tool Calling](/docs/concepts/tool_calling)**: Tool calling is the process of invoking a tool from a chat model where the model requests the execution of a function with specific inputs. **Terms**: [tool-creation](/docs/concepts/tool_calling#tool-creation), [tool-binding](/docs/concepts/tool_calling#tool-binding), [tool-calling](/docs/concepts/tool_calling#tool-calling), [tool-execution](/docs/concepts/tool_calling#tool-execution).

- **[Structured Output](/docs/concepts/structured_outputs)**: A technique that ensures the chat model responds in a structured format, such as JSON, matching a specific schema. **Terms**: [schema-definition](/docs/concepts/structured_outputs#schema-definition), [returning-structured-output](/docs/concepts/structured_outputs#returning-structured-output), [JSON model](/docs/concepts/structured_outputs#json-mode), [with_structured_output()](/docs/concepts/structured_outputs#structured-output-method).

- **[Multimodality](/docs/concepts/multimodality)**: The ability to process data in different forms, such as text, audio, images, and video. **Terms**: [chat-models](/docs/concepts/multimodality#chat-models), [embedding-models](/docs/concepts/multimodality#embedding-models), [vector-stores](/docs/concepts/multimodality#vector-stores).

- **[Tokens](/docs/concepts/tokens)**: Modern large language models process a sequence of units called tokens, which are the fundamental elements used to generate input and output. **Terms**: None.

- **[Runnable Interface](/docs/concepts/runnables)**: The foundational interface for working with LangChain components, implemented across language models, output parsers, retrievers, compiled LangGraph graphs, and more. **Terms**: invoke, batch, stream, astream_events, [input and output types](/docs/concepts/runnables#input-and-output-types), [with_types](/docs/concepts/runnables#with_types), [RunnableConfig](/docs/concepts/runnables#runnableconfig), [Propagation of RunnableConfig](/docs/concepts/runnables#propagation-runnableconfig), [Configurable Runnables](/docs/concepts/runnables#configurable-runnables).

- **[LangChain Expression Language (LCEL)](/docs/concepts/lcel)**: A declarative approach to building new Runnables from existing Runnables. **Terms**: [RunnableSequence](/docs/concepts/lcel#runnablesequence), [RunnableParallel](/docs/concepts/lcel#runnableparallel), The `|` operator (pipe operator).

- **[Retrieval](/docs/concepts/retrieval)**: A mechanism to retrieve relevant documents from a knowledge base in response to a query.

- **[Text Splitters](/docs/concepts/text_splitters)**: Components that split text into chunks for efficient processing.

- **[Embedding Models](/docs/concepts/embedding_models)**: Models that represent data in vector spaces, embedding text, images, and audio into vectors.

- **[VectorStores](/docs/concepts/vectorstores)**: Datastores that store embeddings and support efficient vector searches.

- **[Retriever](/docs/concepts/retrievers)**: A component that retrieves relevant documents from a knowledge base based on a query. **Terms**: [LangChain's Retriever Interface](/docs/concepts/retrievers#interface).

- **[Retrieval Augmented Generation (RAG)](/docs/concepts/rag)**: A technique that enhances language models by combining them with external knowledge bases.

- **[Agents](/docs/concepts/agents)**: Components that combine reasoning and action, often used in multi-step AI systems.

## Additional Concepts

- **[Asynchronous Programming with Langchain](/docs/concepts/async)**: Guidelines for working with LangChain in an asynchronous context.

- **[Callbacks](/docs/concepts/callbacks)**: The callback system in LangChain, consisting of `CallbackManagers` and `CallbackHandlers`, is used for streaming outputs and tracking progress.

## Outdated Concepts

- **[Output Parsers](/docs/concepts/output_parsers)**: Responsible for transforming model output for downstream tasks, primarily used before tool calling and structured outputs became standard.
