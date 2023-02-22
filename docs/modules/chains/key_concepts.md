# Key Concepts

## Chains
A chain is made up of links, which can be either primitives or other chains. 
They vary greatly in complexity and are combination of generic, highly configurable pipelines and more narrow (but usually more complex) pipelines.

## Sequential Chain
This is a specific type of chain where multiple other chains are run in sequence, with the outputs being added as inputs
to the next. A subtype of this type of chain is the [`SimpleSequentialChain`](./generic/sequential_chains.html#simplesequentialchain), where all subchains have only one input and one output,
and the output of one is therefore used as sole input to the next chain.

