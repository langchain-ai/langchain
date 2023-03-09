# Key Concepts

## Agents
Agents use an LLM to determine which actions to take and in what order.
For more detailed information on agents, and different types of agents in LangChain, see [this documentation](agents.md).

## Tools
Tools are functions that agents can use to interact with the world.
These tools can be generic utilities (e.g. search), other chains, or even other agents.
For more detailed information on tools, and different types of tools in LangChain, see [this documentation](tools.md).

## ToolKits
Toolkits are groups of tools that are best used together.
They allow you to logically group and initialize a set of tools that share a particular resource (such as a database connection or json object). 
They can be used to construct an agent for a specific use-case.
