# VW in a langchain chain

Install `requirements.txt`

[VowpalWabbit](https://github.com/VowpalWabbit/vowpal_wabbit)

There is an example notebook (rl_chain.ipynb) with basic usage of the chain.

TLDR:

- Chain is initialized and creates a Vowpal Wabbit instance - only Contextual Bandits and Slates are supported for now
- You can change the arguments at chain creation time
- There is a default prompt but it can be changed
- There is a default reward function that gets triggered and triggers learn automatically
  - This can be turned off and score can be spcified explicitly

Flow:

- Developer: creates chain
- Developer: sets actions
- Developer: calls chain with context and other prompt inputs
- Chain: calls VW with the context and selects an action
- Chain: action (and other vars) are passed to the LLM with the prompt
- Chain: if default reward set, the LLM is called to judge and give a reward score of the response based on the context
- Chain: VW learn is triggered with that score
