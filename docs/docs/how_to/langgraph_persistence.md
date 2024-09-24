# Why LangGraph Persistence?

As of the v0.3 release of LangChain, we recommend that LangChain users
take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
to incorporate `memory` into their LangChain application.

The concept of memory has evolved significantly in LangChain since its initial release.

In LangChain 0.0.x, memory was based on the [BaseMemory](https://api.python.langchain.com/en/latest/memory/langchain_core.memory.BaseMemory.html) interface. 





The main advantage of this interface was that it was easy to plug into other interfaces like `LLMChain` and `AgentExecutor`, and
was useful for prototyping in a non-production setting. The 

While the LangChain 0.0.x memory abstractions were useful, they were limited in their capabilities and not well suited for real-world conversational AI applications. These memory abstractions lacked built-in support for multi-user, multi-conversation scenarios, which are essential for practical conversational AI systems.


[BaseChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html#langchain_core.chat_history.BaseChatMessageHistory) abstractions.


If you are relying on any deprecated memory abstractions in LangChain 0.0.x, we recommend that you follow
the given steps to upgrade to the new LangGraph persistence feature in LangChain 0.3.x.

If you are relying on any of the memory abstractions in LangChain 0.0.x, we recommend that you upgrade to the new LangGraph persistence feature in LangChain 0.3.x.


As of LangChain v0.1, we started recommending that users rely primarily on `BaseChatMessageHistory` for simply
keeping track of the chat history. 

`BaseMemory` has been still available, 
, and `BaseMemory` for more advanced memory management.


## RunnableWithMessageHistory

There is no need to change your code if you are using `RunnableWithMessageHistory`. We do not plan
on deprecating this functionality in the near future. It works for simple chat applications and
any code that uses `RunnableWithMessageHistory` will continue to work as expected.





https://python.langchain.com/docs/versions/migrating_memory/


We've been v




As of v0.3 release, LangChain introduces a new feature: langgraph persistence. This feature allows users to save and load langgraphs to and from disk. This guide will walk you through the process of upgrading your code to use this new feature.