Agents
======

The examples here are all end-to-end agents for specific applications.
In all examples there is an Agent with a particular set of tools.

- Tools: A tool can be anything that takes in a string and returns a string. This means that you can use both the primitives AND the chains found in `this <chains.rst>`_ documentation.
- Agents: An agent uses an LLMChain to determine which tools to use. For a list of all available agent types, see `here <../explanation/agents.md>`_.

**MRKL**

- **Tools used**: Search, SQLDatabaseChain, LLMMathChain
- **Agent used**: `zero-shot-react-description`
- `Paper <https://arxiv.org/pdf/2205.00445.pdf>`_
- **Note**: This is the most general purpose example, so if you are looking to use an agent with arbitrary tools, please start here.
- `Example Notebook <agents/mrkl.ipynb>`_

**Self-Ask-With-Search**

- **Tools used**: Search
- **Agent used**: `self-ask-with-search`
- `Paper <https://ofir.io/self-ask.pdf>`_
- `Example Notebook <agents/self_ask_with_search.ipynb>`_

**ReAct**

- **Tools used**: Wikipedia Docstore
- **Agent used**: `react-docstore`
- `Paper <https://arxiv.org/pdf/2210.03629.pdf>`_
- `Example Notebook <agents/react.ipynb>`_



Additionally, we also provide examples for how to do more customizability:

**Custom Agent**

- Purpose: How to create custom agents.
- `Example Notebook <agents/custom_agent.ipynb>`_


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   agents/*