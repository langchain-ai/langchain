# Agents

Agents use an LLM to determine which actions to take and in what order.
An action can either be using a tool and observing its output, or returning to the user.
For a list of easily loadable tools, see [here](tools.md).
Here are the agents available in LangChain.

For a tutorial on how to load agents, see [here](getting_started.ipynb).

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
