How-To Guides
=============

The first category of how-to guides here cover specific parts of working with agents.

`Custom Tools <examples/custom_tools.ipynb>`_: How to create custom tools that an agent can use.

`Intermediate Steps <examples/intermediate_steps.ipynb>`_: How to access and use intermediate steps to get more visibility into the internals of an agent.

`Custom Agent <examples/custom_agent.ipynb>`_: How to create a custom agent (specifically, a custom LLM + prompt to drive that agent).

`Multi Input Tools <examples/multi_input_tool.ipynb>`_: How to use a tool that requires multiple inputs with an agent.


The next set of examples are all end-to-end agents for specific applications.
In all examples there is an Agent with a particular set of tools.

- Tools: A tool can be anything that takes in a string and returns a string. This means that you can use both the primitives AND the chains found in `this <chains.rst>`_ documentation. LangChain also provides a list of easily loadable tools. For detailed information on those, please see `this documentation <../explanation/tools.md>`_
- Agents: An agent uses an LLMChain to determine which tools to use. For a list of all available agent types, see `here <../explanation/agents.md>`_.

**MRKL**

- **Tools used**: Search, SQLDatabaseChain, LLMMathChain
- **Agent used**: `zero-shot-react-description`
- `Paper <https://arxiv.org/pdf/2205.00445.pdf>`_
- **Note**: This is the most general purpose example, so if you are looking to use an agent with arbitrary tools, please start here.
- `Example Notebook <examples/mrkl.ipynb>`_

**Self-Ask-With-Search**

- **Tools used**: Search
- **Agent used**: `self-ask-with-search`
- `Paper <https://ofir.io/self-ask.pdf>`_
- `Example Notebook <examples/self_ask_with_search.ipynb>`_

**ReAct**

- **Tools used**: Wikipedia Docstore
- **Agent used**: `react-docstore`
- `Paper <https://arxiv.org/pdf/2210.03629.pdf>`_
- `Example Notebook <examples/react.ipynb>`_



.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   examples/*