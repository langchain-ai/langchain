# Key Concepts

## LLMs
Wrappers around Large Language Models (in particular, the "generate" ability of large language models) are at the core of LangChain functionality.
The core method that these classes expose is a `generate` method, which takes in a list of strings and returns an LLMResult (which contains outputs for all input strings).
Read more about LLMResult. This interface operates over a list of strings because often the lists of strings can be batched to the LLM provider,
providing speed and efficiency gains.
For convenience, this class also exposes a simpler, more user friendly interface (via `__call__`). 
The interface for this takes in a single string, and returns a single string.

## Generation
The output of a single generation. Currently in LangChain this is just the generated text, although could be extended in the future
to contain log probs or the like.

## LLMResult
The full output of a call to the `generate` method of the LLM class.
Since the `generate` method takes as input a list of strings, this returns a list of results.
Each result consists of a list of generations (since you can request N generations per input string).
This also contains a `llm_output` attribute which contains provider-specific information about the call.
