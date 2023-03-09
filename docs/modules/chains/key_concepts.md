# Key Concepts

## Chains
A chain is made up of links, which can be either primitives or other chains. 
They vary greatly in complexity and are combination of generic, highly configurable pipelines and more narrow (but usually more complex) pipelines.

## Sequential Chain
This is a specific type of chain where multiple other chains are run in sequence, with the outputs being added as inputs
to the next. A subtype of this type of chain is the [`SimpleSequentialChain`](./generic/sequential_chains.html#simplesequentialchain), where all subchains have only one input and one output,
and the output of one is therefore used as sole input to the next chain.

## Prompt Selectors
One thing that we've noticed is that the best prompt to use is really dependent on the model you use.
Some prompts work really good with some models, but not great with others.
One of our goals is provide good chains that "just work" out of the box. 
A big part of chains like that is having prompts that "just work".
So rather than having a default prompt for chains, we are moving towards a paradigm where if a prompt is not explicitly
provided we select one with a PromptSelector. This class takes in the model passed in, and returns a default prompt.
The inner workings of the PromptSelector can look at any aspect of the model - LLM vs ChatModel, OpenAI vs Cohere, GPT3 vs GPT4, etc.
Due to this being a newer feature, this may not be implemented for all chains, but this is the direction we are moving.
