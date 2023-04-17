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

- [Baby AGI](agents/baby_agi.ipynb): a notebook implementing [BabyAGI](https://github.com/yoheinakajima/babyagi) by Yohei Nakajima as LLM Chains
- [Baby AGI with Tools](agents/baby_agi_with_agent.ipynb): building off the above notebook, this example substitutes in an agent with tools as the execution tools, allowing it to actually take actions.
- [CAMEL](agents/camel_role_playing.ipynb): an implementation of the CAMEL (Communicative Agents for “Mind” Exploration of Large Scale Language Model Society) paper, where two agents communicate with eachother.
- [AI Plugins](agents/custom_agent_with_plugin_retrieval.ipynb): an implementation of an agent that is designed to be able to use all AI Plugins.
- [Generative Agents](agents/characters.ipynb): This notebook implements a generative agent based on the paper [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) by Park, et. al.
