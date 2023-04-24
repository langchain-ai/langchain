# Personal Assistants (Agents)

> [Conceptual Guide](https://docs.langchain.com/docs/use-cases/personal-assistants)


We use "personal assistant" here in a very broad sense.
Personal assistants have a few characteristics:

- They can interact with the outside world
- They have knowledge of your data
- They remember your interactions

Really all of the functionality in LangChain is relevant for building a personal assistant.
Highlighting specific parts:

- [Agent Documentation](../modules/agents.rst) (for interacting with the outside world)
- [Index Documentation](../modules/indexes.rst) (for giving them knowledge of your data)
- [Memory](../modules/memory.rst) (for helping them remember interactions)

Specific examples of this include:

- [AI Plugins](agents/custom_agent_with_plugin_retrieval.ipynb): an implementation of an agent that is designed to be able to use all AI Plugins.
- [Wikibase Agent](agents/wikibase_agent.ipynb): an implementation of an agent that is designed to interact with Wikibase.
