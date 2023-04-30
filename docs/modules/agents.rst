Agents
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/agents>`_


Some applications will require not just a predetermined chain of calls to LLMs/other tools,
but potentially an unknown chain that depends on the user's input.
In these types of chains, there is a “agent” which has access to a suite of tools.
Depending on the user input, the agent can then decide which, if any, of these tools to call.

In this section of documentation, we first start with a Getting Started notebook to cover how to use all things related to agents in an end-to-end manner.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./agents/getting_started.ipynb


We then split the documentation into the following sections:

**Tools**

An overview of the various tools LangChain supports.


**Agents**

An overview of the different agent types.


**Toolkits**

An overview of toolkits, and examples of the different ones LangChain supports.


**Agent Executor**

An overview of the Agent Executor class and examples of how to use it.

Go Deeper
---------

.. toctree::
   :maxdepth: 1

   ./agents/tools.rst
   ./agents/agents.rst
   ./agents/toolkits.rst
   ./agents/agent_executors.rst
