# Autonomous Agents


Autonomous Agents are agents that designed to be more long running.
You give them one or multiple long term goals, and they independently execute towards those goals.
The applications combine tool usage and long term memory.

At the moment, Autonomous Agents are fairly experimental and based off of other open-source projects.
By implementing these open source projects in LangChain primitives we can get the benefits of LangChain - 
easy switching an experimenting with multiple LLMs, usage of different vectorstores as memory, 
usage of LangChain's collection of tools.

## Baby AGI ([Original Repo](https://github.com/yoheinakajima/babyagi))

- [Baby AGI](autonomous_agents/baby_agi.ipynb): a notebook implementing BabyAGI as LLM Chains
- [Baby AGI with Tools](autonomous_agents/baby_agi_with_agent.ipynb): building off the above notebook, this example substitutes in an agent with tools as the execution tools, allowing it to actually take actions.


## AutoGPT ([Original Repo](https://github.com/Significant-Gravitas/Auto-GPT))
- [AutoGPT](autonomous_agents/autogpt.ipynb): a notebook implementing AutoGPT in LangChain primitives
- [WebSearch Research Assistant](autonomous_agents/marathon_times.ipynb): a notebook showing how to use AutoGPT plus specific tools to act as research assistant that can use the web.
