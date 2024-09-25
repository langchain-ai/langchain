# How to upgrade to LangGraph persistence

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into their LangChain application.

This how-to guide will contain links to other relevant how-to guides for specific situations.

## How-tos

## Evolution of memory in LangChain 

The concept of memory has evolved significantly in LangChain since its initial release.

In LangChain 0.0.x, memory was based on the [BaseMemory](https://api.python.langchain.com/en/latest/memory/langchain_core.memory.BaseMemory.html) interface, and partially on the [BaseChatMessageHistory](https://api.python.langchain.com/en/latest/history/langchain_core.runnables.history.BaseChatMessageHistory.html) interface.

There were a number of [memory implementations](https://python.langchain.com/api_reference/langchain/memory.html) based
on the `BaseMemory` interface, such as [ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html), [ConversationBufferWindowMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html) etc. 

While these implementations proved useful for prototyping, they were limited in their capability and not well suited for real-world conversational AI applications. The abstractions lacked built-in support for multi-user, multi-conversation scenarios, which are essential for practical conversational AI systems.

:::note
If you are relying on any deprecated memory abstractions in LangChain 0.0.x, we recommend that you follow
the given steps to upgrade to the new LangGraph persistence feature in LangChain 0.3.x.
https://python.langchain.com/docs/versions/migrating_memory/
:::

As of LangChain v0.1, we started recommending that users rely primarily on [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory). 

`BaseChatMessageHistory` 

At this stage, the only option for orchestrating LangChain chains was via [LCEL](https://python.langchain.com/docs/how_to/#langchain-expression-language-lcel). So users were expected to use `BaseChatMessageHistory` in conjunction with `LCEL` to manage memory in LangChain via the
[RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory) interface.

While this option is more than sufficient for a simple chat application, many users found the API
to be unintuitive and difficult to work with.

 
As of LangChain v0.3, we are recommending that any new code gets written with new LangGraph persistence in mind.

:::important
If you have been using `RunnableWithMessageHistory` in your code, you do not need to make any changes. We do not plan on deprecating this functionality in the near future. This functionality is sufficient for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.
:::
