# Conceptual Guide

## Overview

In this guide, you'll find explanations of the key concepts, providing a deeper understanding of core principles.

We recommend that you go through at least one of the [Tutorials](/docs/tutorials) before diving into the conceptual guide. This will help you understand the context and practical applications of the concepts discussed here.

The conceptual guide will not cover step-by-step instructions or specific implementation details — those are found in the [How-To Guides](/docs/how_to/) and [Tutorials](/docs/tutorials) sections. For detailed reference material, please visit the [API Reference](https://python.langchain.com/api_reference/).

## Concepts

- **[Why LangChain?](/docs/concepts/why_langchain)**: Why LangChain is the best choice for building AI applications.

- **[Architecture](/docs/concepts/architecture)**: Overview of how packages are organized in the LangChain ecosystem.

- **[Chat Models](/docs/concepts/chat_models)**: Modern LLMs exposed via a chat interface which process sequences of messages as input and output a message.

- **[Messages](/docs/concepts/messages)**: Messages are the unit of communication in modern LLMs, used to represent input and output of a chat model, as well as any additional context or metadata that may be associated with the conversation.

- **[Chat History](/docs/concepts/chat_history)**: Chat history is a record of the conversation between the user and the chat model, used to maintain context and state throughout the conversation.

- **[Tools](/docs/concepts/tools)**: The **tool** abstraction in LangChain associates a Python **function** with a **schema** defining the function's **name**, **description**, and **input**.

- **[Tool Calling](/docs/concepts/tool_calling)**: Tool calling is the process of invoking a tool from a chat model.

- **[Structured Output](/docs/concepts/structured_outputs)**: A technique to make the chat model respond in a structured format, such as JSON and matching a specific schema.

- **[Multimodality](/docs/concepts/multimodality)**: The ability to work with data that comes in different forms, such as text, audio, images, and video.

- **[Tokens](/docs/concepts/tokens)**: Modern large language models (LLMs) are typically based on a transformer architecture that processes a sequence of units known as tokens.

- **[Runnable Interface](/docs/concepts/runnables)**: The Runnable interface is foundational for working with LangChain components.

- **[LangChain Expression Language (LCEL)](/docs/concepts/lcel)**: A declarative approach to building new Runnables from existing Runnables.

- **[Retrieval](/docs/concepts/retrieval)**

- **[Text Splitters](/docs/concepts/text_splitters)**

- **[Embedding Models](/docs/concepts/embedding_models)**: Embedding models are models that can represent data in a vector space.

- **[VectorStores](/docs/concepts/vectorstores)**: A datastore that can store embeddings and associated data and supports efficient vector search.

- **[Retriever](/docs/concepts/retrievers)**: A retriever is a component that retrieves relevant documents from a knowledge base in response to a query.

- **[Retrieval Augmented Generation (RAG)](/docs/concepts/rag)**: A powerful technique that enhances language models by combining them with external knowledge bases.

- **[Agents](/docs/concepts/agents)**

## Glossary

- **[langchain](/docs/concepts/architecture#langchain)**: Core package of LangChain.
- **[langchain-community](/docs/concepts/architecture#langchain-community)**: Community-driven components for LangChain.
- **[langchain-core](/docs/concepts/architecture#langchain-core)**: Core infrastructure for building with LangChain.
- **[langserve](/docs/concepts/architecture#langserve)**: Deployment and serving platform for LangChain.
- **[langgraph](/docs/concepts/architecture#langgraph)**: A graph-based execution engine in LangChain.
- **[partner-packages](/docs/concepts/architecture#partner-packages)**: Third-party packages that integrate with LangChain.
- **[key methods](/docs/concepts/chat_models#key-methods)**: Key methods like `invoke`, `batch`, `stream`, and `bind_tools`, `with_structured_output`.
- **[standard parameters](/docs/concepts/chat_models#standard-parameters)**: Parameters such as API key, temperature, and max_tokens.
- **[context-window](/docs/concepts/chat_models#context-window)**: The span of tokens a model can process.
- **[with-structured-output](/docs/concepts/chat_models#with-structured-output)**: A method for generating structured responses.
- **[bind-tools](/docs/concepts/chat_models#bind-tools)**: Allows models to interact with tools.
- **[multimodality](/docs/concepts/chat_models#multimodality)**: Capability to process different types of data like text, audio, and images.
- **[rate-limiting](/docs/concepts/chat_models#rate-limiting)**: Mechanism to control the flow of requests to avoid overloading.
- **[caching](/docs/concepts/chat_models#caching)**: Storing results to avoid recomputation.
- **[role](/docs/concepts/messages#role)**: Represents the role (e.g., user, assistant) in a message.
- **[content](/docs/concepts/messages#content)**: The message text itself.
- **[HumanMessage](/docs/concepts/messages#humanmessage)**: Represents a message from a human user.
- **[AIMessage](/docs/concepts/messages#aimessage)**: Represents a message from an AI model.
- **[ToolMessage](/docs/concepts/messages#toolmessage)**: Represents a message that invokes a tool.
- **[AIMessageChunk](/docs/concepts/messages#aimessagechunk)**: A partial response from an AI message.
- **[RemoveMessage](/docs/concepts/messages#remove-message)**: A method for deleting a message.
- **[OpenAI format](/docs/concepts/messages#openai-format)**: OpenAI's message format used for communication.
- **[conversation-patterns](/docs/concepts/chat_history#conversation-patterns)**: Common patterns in chat interactions.
- **[managing-chat-history](/docs/concepts/chat_history#managing-chat-history)**: Techniques to maintain and manage the chat history.
- **[@tool](/docs/concepts/tools#@tool)**: Decorator for defining tools in LangChain.
- **[BaseTool](/docs/concepts/tools#basetool)**: The base class for all tools in LangChain.
- **[Tool-artifacts](/docs/concepts/tools#tool-artifacts)**: Artifacts produced by tools.
- **[InjectedToolArg](/docs/concepts/tools#injectedtoolarg)**: Mechanism to inject arguments into tool functions.
- **[RunnableConfig](/docs/concepts/tools#runnableconfig)**: Configuration for runnables and tools.
- **[InjectedState](/docs/concepts/tools#injectedstate)**: A state injected into a tool function.
- **[InjectedStore](/docs/concepts/tools#injectedstore)**: A store that can be injected into a tool for data persistence.
- **[tool-creation](/docs/concepts/tool_calling#tool-creation)**: Process of creating tools for tool calling.
- **[tool-binding](/docs/concepts/tool_calling#tool-binding)**: Binding tools to models.
- **[tool-calling](/docs/concepts/tool_calling#tool-calling)**: Invoking tools within models.
- **[tool-execution](/docs/concepts/tool_calling#tool-execution)**: Executing a tool during a tool call.
- **[schema-definition](/docs/concepts/structured_outputs#schema-definition)**: Defining a schema for structured outputs.
- **[returning-structured-output](/docs/concepts/structured_outputs#returning-structured-output)**: Returning output in a structured format.
- **[JSON model](/docs/concepts/structured_outputs#json-mode)**: Returning responses in JSON format.
- **[with_structured_output()](/docs/concepts/structured_outputs#structured-output-method)**: Method to enforce structured output.
- **[chat-models](/docs/concepts/multimodality#chat-models)**: Chat models that handle multiple data modalities.
- **[embedding-models](/docs/concepts/multimodality#embedding-models)**: Models that generate vector embeddings for various data types.
- **[vector-stores](/docs/concepts/multimodality#vector-stores)**: Datastores for storing and searching vector embeddings.
- **[input and output types](/docs/concepts/runnables#input-and-output-types)**: Types used for input and output in runnables.
- **[with_types](/docs/concepts/runnables#with_types)**: Method to enforce input/output types.
- **[RunnableConfig](/docs/concepts/runnables#runnableconfig)**: Configuration for runnables.
- **[Propagation of RunnableConfig](/docs/concepts/runnables#propagation-runnableconfig)**: Propagating configuration through runnables.
- **[Configurable Runnables](/docs/concepts/runnables#configurable-runnables)**: Runn
