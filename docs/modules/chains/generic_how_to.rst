Generic Chains
--------------

A chain is made up of links, which can be either primitives or other chains.
Primitives can be either `prompts <../prompts.html>`_, `llms <../llms.html>`_, `utils <../utils.html>`_, or other chains.
The examples here are all generic end-to-end chains that are meant to be used to construct other chains rather than serving a specific purpose.

**LLMChain**

- **Links Used**: PromptTemplate, LLM
- **Notes**: This chain is the simplest chain, and is widely used by almost every other chain. This chain takes arbitrary user input, creates a prompt with it from the PromptTemplate, passes that to the LLM, and then returns the output of the LLM as the final output.
- `Example Notebook <./generic/llm_chain.html>`_

**Transformation Chain**

- **Links Used**: TransformationChain
- **Notes**: This notebook shows how to use the Transformation Chain, which takes an arbitrary python function and applies it to inputs/outputs of other chains.
- `Example Notebook <./generic/transformation.html>`_

**Sequential Chain**

- **Links Used**: Sequential
- **Notes**: This notebook shows how to combine calling multiple other chains in sequence.
- `Example Notebook <./generic/sequential_chains.html>`_

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Generic Chains
   :name: generic
   :hidden:

   ./generic/*