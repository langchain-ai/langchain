"""Chain for applying self-critique using the SmartGPT workflow.

See details at https://youtu.be/wVzuvf9D9BU

The workflow performs these 3 steps:
1. **Ideate**: Pass the user prompt to an ideation LLM n_ideas times,
   each result is an "idea"
2. **Critique**: Pass the ideas to a critique LLM which looks for flaws in the ideas
   & picks the best one
3. **Resolve**: Pass the critique to a resolver LLM which improves upon the best idea
   & outputs only the (improved version of) the best output

In total, the SmartGPT workflow will use n_ideas+2 LLM calls

Note that SmartLLMChain will only improve results (compared to a basic LLMChain),
when the underlying models have the capability for reflection, which smaller models
often don't.

Finally, a SmartLLMChain assumes that each underlying LLM outputs exactly 1 result.
"""

from langchain_experimental.smart_llm.base import SmartLLMChain

__all__ = ["SmartLLMChain"]
