Agents
==========================

Some applications will require not just a predetermined chain of calls to LLMs/other tools,
but potentially an unknown chain that depends on the user's input.
In these types of chains, there is a “agent” which has access to a suite of tools.
Depending on the user input, the agent can then decide which, if any, of these tools to call.

In this section of documentation, we first start with a Getting Started notebook to over over how to use all things related to agents in an end-to-end manner.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./agents/getting_started.ipynb


We then split the documentation into the following sections:

**Tools**

An overview of the various tools LangChain supports.

.. toctree::
   :maxdepth: 1

   ./agents/tools.rst

**Agents**

An overview of the different agent types.

.. toctree::
   :maxdepth: 1

   ./agents/agents.rst


**Toolkits**

An overview of toolkits, and examples of the different ones LangChain supports.

.. toctree::
   :maxdepth: 1

   ./agents/agent_toolkits.rst


**Agent Executor**

An overview of the Agent Executor class and examples of how to use it.

.. toctree::
   :maxdepth: 1

   ./agents/agent_executors.rst
