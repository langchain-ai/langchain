# Agents

There are three types of examples in this section:

1. Agent Overview: how-to-guides for generic agent functionality
2. Agent Toolkits: how-to-guides for specific agent toolkits (agents optimized for interacting with a certain resource)
3. Agent Types: how-to-guides for working with the different agent types

## Agent Overview

The first category of how-to guides here cover specific parts of working with agents.

`Load From Hub <./examples/load_from_hub.html>`_: This notebook covers how to load agents from `LangChainHub <https://github.com/hwchase17/langchain-hub>`_.

`Custom Tools <./examples/custom_tools.html>`_: How to create custom tools that an agent can use.

`Agents With Vectorstores <./examples/agent_vectorstore.html>`_: How to use vectorstores with agents.

`Intermediate Steps <./examples/intermediate_steps.html>`_: How to access and use intermediate steps to get more visibility into the internals of an agent.

`Custom Agent <./examples/custom_agent.html>`_: How to create a custom agent (specifically, a custom LLM + prompt to drive that agent).

`Multi Input Tools <./examples/multi_input_tool.html>`_: How to use a tool that requires multiple inputs with an agent.

`Search Tools <./examples/search_tools.html>`_: How to use the different type of search tools that LangChain supports.

`Max Iterations <./examples/max_iterations.html>`_: How to restrict an agent to a certain number of iterations.

`Asynchronous <./examples/async_agent.html>`_: Covering asynchronous functionality.


## Agent Types

The final set of examples are all end-to-end example of different agent types.
In all examples there is an Agent with a particular set of tools.

- Tools: A tool can be anything that takes in a string and returns a string. This means that you can use both the primitives AND the chains found in `this <../chains.html>`_ documentation. LangChain also provides a list of easily loadable tools. For detailed information on those, please see `this documentation <./tools.html>`_
- Agents: An agent uses an LLMChain to determine which tools to use. For a list of all available agent types, see `here <./agents.html>`_.

**MRKL**

- **Tools used**: Search, SQLDatabaseChain, LLMMathChain
- **Agent used**: `zero-shot-react-description`
- `Paper <https://arxiv.org/pdf/2205.00445.pdf>`_
- **Note**: This is the most general purpose example, so if you are looking to use an agent with arbitrary tools, please start here.
- `Example Notebook <./implementations/mrkl.html>`_

**Self-Ask-With-Search**

- **Tools used**: Search
- **Agent used**: `self-ask-with-search`
- `Paper <https://ofir.io/self-ask.pdf>`_
- `Example Notebook <./implementations/self_ask_with_search.html>`_

**ReAct**

- **Tools used**: Wikipedia Docstore
- **Agent used**: `react-docstore`
- `Paper <https://arxiv.org/pdf/2210.03629.pdf>`_
- `Example Notebook <./implementations/react.html>`_

## Agent Types

Agents use an LLM to determine which actions to take and in what order.
An action can either be using a tool and observing its output, or returning a response to the user.
Here are the agents available in LangChain.

## `zero-shot-react-description`

This agent uses the ReAct framework to determine which tool to use
based solely on the tool's description. Any number of tools can be provided.
This agent requires that a description is provided for each tool.

## `react-docstore`

This agent uses the ReAct framework to interact with a docstore. Two tools must
be provided: a `Search` tool and a `Lookup` tool (they must be named exactly as so).
The `Search` tool should search for a document, while the `Lookup` tool should lookup
a term in the most recently found document.
This agent is equivalent to the
original [ReAct paper](https://arxiv.org/pdf/2210.03629.pdf), specifically the Wikipedia example.

## `self-ask-with-search`

This agent utilizes a single tool that should be named `Intermediate Answer`.
This tool should be able to lookup factual answers to questions. This agent
is equivalent to the original [self ask with search paper](https://ofir.io/self-ask.pdf),
where a Google search API was provided as the tool.

### `conversational-react-description`

This agent is designed to be used in conversational settings.
The prompt is designed to make the agent helpful and conversational.
It uses the ReAct framework to decide which tool to use, and uses memory to remember the previous conversation interactions.
