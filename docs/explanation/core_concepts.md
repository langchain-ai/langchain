# Core Concepts

This section goes over the core concepts of LangChain.
Understanding these will go a long way in helping you understand the codebase and how to construct chains.

## PromptTemplates
PromptTemplates generically have a `format` method that takes in variables and returns a formatted string.
The most simple implementation of this is to have a template string with some variables in it, and then format it with the incoming variables.
More complex iterations dynamically construct the template string from few shot examples, etc.

For a more detailed explanation of how LangChain approaches prompts and prompt templates, see [here](/examples/prompts/prompt_management).

## LLMs
Wrappers around Large Language Models (in particular, the `generate` ability of large language models) are some of the core functionality of LangChain.
These wrappers are classes that are callable: they take in an input string, and return the generated output string.

## Embeddings
These classes are very similar to the LLM classes in that they are wrappers around models, 
but rather than return a string they return an embedding (list of floats). This are particularly useful when 
implementing semantic search functionality. They expose separate methods for embedding queries versus embedding documents.

## Vectorstores
These are datastores that store documents. They expose a method for passing in a string and finding similar documents.

## Chains
These are pipelines that combine multiple of the above ideas. 
They vary greatly in complexity and are combination of generic, highly configurable pipelines and more narrow (but usually more complex) pipelines.

## Agents
As opposed to a chain, whether the steps to be taken are known ahead of time, agents
use an LLM to determine which tools to call and in what order.

## Memory
By default, Chains and Agents are stateless, meaning that they treat each incoming query independently.
In some applications (chatbots being a GREAT example) it is highly important to remember previous interactions,
both at a short term but also at a long term level. The concept of "Memory" exists to do exactly that.

