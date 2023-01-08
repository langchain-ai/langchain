# Key Concepts

## Chains
A chain is made up of links, which can be either primitives or other chains. 
They vary greatly in complexity and are combination of generic, highly configurable pipelines and more narrow (but usually more complex) pipelines.

## Sequential Chain
This is a specific type of chain where multiple other chains are run in sequence, with the outputs being added as inputs
to the next. A subtype of this type of chain is the `SimpleSequentialChain`, where all subchains have only one input and one output,
and the output of one is therefore used as sole input to the next chain.

## CombineDocuments Chains
These are a subset of chains designed to work with documents. There are two pieces to consider:

1. The underlying chain method (eg, how the documents are combined)
2. Use cases for these types of chains.

For the first, please see [this documentation](combine_docs.md) for more detailed information on the types of chains LangChain supports.
For the second, please see the Use Cases section for more information on [question answering](/use_cases/question_answering.md), 
[question answering with sources](/use_cases/qa_with_sources.md), and [summarization](/use_cases/summarization.md).

