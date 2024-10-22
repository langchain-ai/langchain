# Conceptual guide

## Overview

In this guide, you'll find explanations of the key concepts, providing a deeper understanding of core principles.

We recommend that you go through at least one of the [Tutorials](/docs/tutorials) before diving into the conceptual guide. This will help you understand the context and practical applications of the concepts discussed here.

The conceptual guide will not cover step-by-step instructions or specific implementation details — those are found in the [How-To Guides](/docs/how_to/) and [Tutorials](/docs/tutorials) sections. For detailed reference material, please visit the [API Reference](https://python.langchain.com/api_reference/).

## Table of Contents

- **[Why LangChain?](/docs/concepts/why_langchain)**: Why LangChain is the best choice for building AI applications.

- **[Architecture](/docs/concepts/architecture)**: Overview of how packages are organized in the LangChain ecosystem.
    - **Terms**:
        - [langchain](/docs/concepts/architecture#langchain)
        - [langchain-community](/docs/concepts/architecture#langchain-community)
        - [langchain-core](/docs/concepts/architecture#langchain-core)
        - [langserve](/docs/concepts/architecture#langserve)
        - [langgraph](/docs/concepts/architecture#langgraph)
        - [partner-packages](/docs/concepts/architecture#partner-packages).

- **[Chat Models](/docs/concepts/chat_models)**: Modern LLMs exposed via a chat interface which process sequences of messages as input and output a message.
    - **Terms**:
        - [key methods](/docs/concepts/chat_models#key-methods): Key methods like `invoke`, `batch`, `stream`, and `bind_tools`, `with_structured_output`.
        - [standard parameters](/docs/concepts/chat_models#standard-parameters): set parameters like API key, temperature, max_tokens, and more.
        - [context-window](/docs/concepts/chat_models#context-window)
        - [with-structured-output](/docs/concepts/chat_models#with-structured-output)
        - [bind-tools](/docs/concepts/chat_models#bind-tools).
        - [multimodality](/docs/concepts/chat_models#multimodality)
        - [rate-limiting](/docs/concepts/chat_models#rate-limiting)
        - [caching](/docs/concepts/chat_models#caching)
         
- **[Messages](/docs/concepts/messages)**: Messages are the unit of communication in modern LLMs, used to represent input and output of a chat model, as well as any additional context or metadata that may be associated with the conversation.
    - **Terms**:
        - [role](/docs/concepts/messages#role)
        - [content](/docs/concepts/messages#content)
        - [HumanMessage](/docs/concepts/messages#humanmessage)
        - [AIMessage](/docs/concepts/messages#aimessage)
        - [ToolMessage](/docs/concepts/messages#toolmessage)
        - [AIMessageChunk](/docs/concepts/messages#aimessagechunk)
        - [RemoveMessage](/docs/concepts/messages#remove-message)
        - [OpenAI format](/docs/concepts/messages#openai-format)

- **[Chat History](/docs/concepts/chat_history)**: Chat history is a record of the conversation between the user and the chat model, used to maintain context and state throughout the conversation. The chat history is a sequence of [messages](/docs/concepts/messages), each associated with a specific [role](/docs/concepts/messages#role) such as "user", "assistant", "system", or "tool".
    - **Terms**:
        - [conversation-patterns](/docs/concepts/chat_history#conversation-patterns)
        - [managing-chat-history](/docs/concepts/chat_history#managing-chat-history)

- **[Tools](/docs/concepts/tools)**: The **tool** abstraction in LangChain associates a Python **function** with a **schema** defining the function's **name**, **description**, and **input**. Tools can be passed to chat models that support [tool calling](/docs/concepts/tool_calling), allowing the model to request the execution of a specific function with specific inputs.
    - **Terms**:
        - [@tool](/docs/concepts/tools#@tool)
        - [BaseTool](/docs/concepts/tools#basetool)
        - [Tool-artifacts](/docs/concepts/tools#tool-artifacts)
        - [InjectedToolArg](/docs/concepts/tools#injectedtoolarg)
        - [RunnableConfig](/docs/concepts/tools#runnableconfig)
        - [InjectedState](/docs/concepts/tools#injectedstate)
        - [InjectedStore](/docs/concepts/tools#injectedstore)
         
- **[Tool Calling](/docs/concepts/tool_calling)**: Tool calling is the process of invoking a tool from a chat model. The model can request the execution of a specific function with specific inputs. The tool is executed, and its output is returned to the model.
    - **Terms**:
        - [tool-creation](/docs/concepts/tool_calling#tool-creation)
        - [tool-binding](/docs/concepts/tool_calling#tool-binding)
        - [tool-calling](/docs/concepts/tool_calling#tool-calling)
        - [tool-execution](/docs/concepts/tool_calling#tool-execution).

- **[Structured Output](/docs/concepts/structured_outputs)**: A technique to make the chat model respond in a structured format, such as JSON and matching a specific schema.
  - **Terms**:
    - [schema-definition](/docs/concepts/structured_outputs#schema-definition)
    - [returning-structured-output](/docs/concepts/structured_outputs#returning-structured-output).
    - [JSON model](/docs/concepts/structured_outputs#json-mode)
    - [with_structured_output()](/docs/concepts/structured_outputs#structured-output-method)
     
- **[Multimodality](/docs/concepts/multimodality)**: The ability to work with data that comes in different forms, such as text, audio, images, and video. Multimodality can appear in various components, allowing models and systems to handle and process a mix of these data types seamlessly.
  - **Terms**:
    - [chat-models](/docs/concepts/multimodality#chat-models)
    - [embedding-models](/docs/concepts/multimodality#embedding-models)
    - [vector-stores](/docs/concepts/multimodality#vector-stores).
 
- **[Tokens](/docs/concepts/tokens)**: Modern large language models (LLMs) are typically based on a transformer architecture that processes a sequence of units known as tokens. Tokens are the fundamental elements that models use to break down input and generate output. 
- **[Runnable Interface](/docs/concepts/runnables)**: The Runnable interface is foundational for working with LangChain components, and it's implemented across many of them, such as [language models](/docs/concepts/chat_models), [output parsers](/docs/concepts/output_parsers), [retrievers](/docs/concepts/retrievers), [compiled LangGraph graphs](
  https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph) and more.
  - **Terms**:
    - `invoke`, `batch`, `stream`, `astream_events`
    - [input and output types](/docs/concepts/runnables#input-and-output-types)
    - [with_types](/docs/concepts/runnables#with_types)
    - [RunnableConfig](/docs/concepts/runnables#runnableconfig)
    - [Propagation of RunnableConfig](/docs/concepts/runnables#propagation-runnableconfig)
    - [Configurable Runnables](/docs/concepts/runnables#configurable-runnables)
- **[LangChain Expression Language (LCEL)](/docs/concepts/lcel)**: a **declarative** approach to building new [Runnables](/docs/concepts/runnables) from existing Runnables.
  - **Terms**:
    - [RunnableSequence](/docs/concepts/lcel#runnablesequence)
    - [RunnableParallel](/docs/concepts/lcel#runnableparallel)
    - The `|` operator (pipe operator) in LCEL.
- **[Retrieval](/docs/concepts/retrieval)**
- **[Text Splitters](/docs/concepts/text_splitters)**
- **[Embedding Models](/docs/concepts/embedding_models)**: Embedding models are models that can represent data in a vector space. They can embed various forms of data, such as text, images, and audio, into vector spaces.
  - **Terms**:
- **[VectorStores](/docs/concepts/vectorstores)**: A datastore that can store embeddings and associated data and supports efficient vector search.
- **[Retriever](/docs/concepts/retriever)**: A retriever is a component that retrieves relevant documents from a knowledge base in response to a query.
  - **Terms**:
    - [LangChain's Retriever Interface](/docs/concepts/retriever#interface)
- **[Retrieval Augmented Generation (RAG)](/docs/concepts/rag)**: A powerful technique that enhances language models by combining them with external knowledge bases.
- **[Agents](/docs/concepts/agents)**
  Retrieval Augmented Generation (RAG) is a powerful technique that enhances [language models](/docs/concepts/chat_models/) by combining them with external knowledge bases.


## Additional Concepts

- [Asynchronous Programming with Langchain](/docs/concepts/async): This guide covers some basic things that one should know to work with LangChain in an asynchronous context.
- [Callbacks](/docs/concepts/callbacks): Learn about the callback system in LangChain. It is composed of `CallbackManagers` (which dispatch events to the registered handlers) and `CallbackHandlers` (which handle the events). Callbacks are used to stream outputs from LLMs in LangChain, observe the progress of an LLM application, and more.

## Outdated Concepts

- [Output Parsers](/docs/concepts/output_parsers): Output parsers are responsible for taking the output of a model and transforming it into a more suitable format for downstream tasks. Output parsers were primarily useful prior to the general availability of chat models that natively support [tool calling](/docs/concepts/tool_calling) and [structured outputs](/docs/concepts/structured_outputs).