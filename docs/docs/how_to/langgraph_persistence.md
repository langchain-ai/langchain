# How to upgrade to LangGraph persistence

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into their LangChain application.

## Evolution of memory in LangChain 

The concept of memory has evolved significantly in LangChain since its initial release.

In LangChain 0.0.x, memory was based on the [BaseMemory](https://api.python.langchain.com/en/latest/memory/langchain_core.memory.BaseMemory.html) interface and the [BaseChatMessageHistory](https://api.python.langchain.com/en/latest/history/langchain_core.runnables.history.BaseChatMessageHistory.html) interface.

There were number of useful [memory implementations](https://python.langchain.com/api_reference/langchain/memory.html) based
on the `BaseMemory` interface (e.g.[ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html), [ConversationBufferWindowMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)); however, these lacked built-in support for multi-user, multi-conversation scenarios, which are essential for practical conversational AI systems.

:::note
If you are relying on any deprecated memory abstractions in LangChain 0.0.x, we recommend that you follow
the given steps to upgrade to the new LangGraph persistence feature in LangChain 0.3.x.
https://python.langchain.com/docs/versions/migrating_memory/
:::

As of LangChain v0.1, we started recommending that users rely primarily on [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory). `BaseChatMessageHistory` is a simple persistence layer for a chat history that can be used to store and retrieve messages in a conversation. At this time, the only option for orchestrating LangChain chains was via [LCEL](https://python.langchain.com/docs/how_to/#langchain-expression-language-lcel). When using `LCEL`, memory can be added using the [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory) interface. While this option is sufficient for building a simple chat application, many users found the API to be unintuitive and difficult to work with.

As of LangChain v0.3, we are commending that new code rely on LangGraph for both orchestration and persistence.

Specifically, for orchestration instead of writing `LCEL` code, users can define LangGraph [graphs](https://langchain-ai.github.io/langgraph/concepts/low_level/). This allows users to keep using `LCEL` within individual nodes when `LCEL` is needed, while
making it easy to define complex orchestration logic that is more readable and maintainable.

For persistence, users can use LangGraph's [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) feature to store and retrieve data from a graph database. LangGraph persistence is extremely flexible and can support a much wider range of use cases than the `RunnableWithMessageHistory` interface.

:::important
If you have been using `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do not need to make any changes. We do not plan on deprecating either functionality in the near future. This functionality is sufficient for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.
:::
