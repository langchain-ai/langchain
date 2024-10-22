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
        - [chatmodel](/docs/concepts/chat_models#chatmodel)
        - [llm](/docs/concepts/chat_models#llm)
        - [caching](/docs/concepts/chat_models#caching)
        - [rate-limiting](/docs/concepts/chat_models#rate-limiting)
        - [context-window](/docs/concepts/chat_models#context-window)
        - [with-structured-output](/docs/concepts/chat_models#with-structured-output)
        - [bind-tools](/docs/concepts/chat_models#bind-tools).

- **[Messages](/docs/concepts/messages)**: Messages are the unit of communication in modern LLMs, used to represent input and output of a chat model, as well as any additional context or metadata that may be associated with the conversation.
    - **Terms**:
        - [role](/docs/concepts/messages#role)
        - [content](/docs/concepts/messages#content)
        - [humanmessage](/docs/concepts/messages#humanmessage)
        - [aimessage](/docs/concepts/messages#aimessage)
        - [toolmessage](/docs/concepts/messages#toolmessage)
        - [aimessagechunk](/docs/concepts/messages#aimessagechunk)
        - [tool-call](/docs/concepts/messages#tool-call)
        - [openai-format](/docs/concepts/messages#openai-format)
        - [remove-message](/docs/concepts/messages#remove-message).

- **[Chat History](/docs/concepts/chat_history)**: Chat history is a record of the conversation between the user and the chat model, used to maintain context and state throughout the conversation. The chat history is a sequence of [messages](/docs/concepts/messages), each associated with a specific [role](/docs/concepts/messages#role) such as "user", "assistant", "system", or "tool".
    - **Terms**:
        - [conversation-patterns](/docs/concepts/chat_history#conversation-patterns)
        - [managing-chat-history](/docs/concepts/chat_history#managing-chat-history).

- **[Tools](/docs/concepts/tools)**: The **tool** abstraction in LangChain associates a Python **function** with a **schema** defining the function's **name**, **description**, and **input**. Tools can be passed to chat models that support [tool calling](/docs/concepts/tool_calling), allowing the model to request the execution of a specific function with specific inputs.
    - **Terms**:
        - [@tool](/docs/concepts/tools#@tool)
        - [basetool](/docs/concepts/tools#basetool)
        - [tool-artifacts](/docs/concepts/tools#tool-artifacts)
        - [injectedtoolarg](/docs/concepts/tools#injectedtoolarg)
        - [runnableconfig](/docs/concepts/tools#runnableconfig)
        - [injectedstate](/docs/concepts/tools#injectedstate)
        - [injectedstore](/docs/concepts/tools#injectedstore).

- **[Tool Calling](/docs/concepts/tool_calling)**: Tool calling is the process of invoking a tool from a chat model. The model can request the execution of a specific function with specific inputs. The tool is executed, and its output is returned to the model.
    - **Terms**:
        - [tool-creation](/docs/concepts/tool_calling#tool-creation)
        - [tool-binding](/docs/concepts/tool_calling#tool-binding)
        - [tool-calling](/docs/concepts/tool_calling#tool-calling)
        - [tool-execution](/docs/concepts/tool_calling#tool-execution).

- [Structured Output](/docs/concepts/structured_outputs): 
- [Multimodality](/docs/concepts/multimodality)
- [Embedding Models](/docs/concepts/embedding_models)
- [VectorStores](/docs/concepts/vectorstores)
- [Runnable Interface](/docs/concepts/runnables)
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
- [LangGraph](/docs/concepts/langgraph)
- [Multimodality](/docs/concepts/multimodality)
- [Tokens](/docs/concepts/tokens)
- [Agents](/docs/concepts/agents)
- [Callbacks](/docs/concepts/callbacks)
- [Text Splitting](/docs/concepts/text_splitting)
- [Structured Output](/docs/concepts/structured_data)
- [Retrieval](/docs/concepts/retrieval)
- [Retriever](/docs/concepts/retriever)
- [Retrieval Augmented Generation (RAG)](/docs/concepts/rag)
- [Asynchronous Programming with Langchain](/docs/concepts/async)
