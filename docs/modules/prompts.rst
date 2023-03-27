Prompts
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/prompts>`_


The new way of programming models is through prompts.
A "prompt" refers to the input to the model.
This input is rarely hard coded, but rather is often constructed from multiple components.
A PromptTemplate is responsible for the construction of this input.
LangChain provides several classes and functions to make constructing and working with prompts easy.

This section of documentation is split into four sections:

**LLM Prompt Templates**

How to use PromptTemplates to prompt Language Models.

**Chat Prompt Templates**

How to use PromptTemplates to prompt Chat Models.

**Example Selectors**

Often times it is useful to include examples in prompts.
These examples can be hardcoded, but it is often more powerful if they are dynamically selected.
This section goes over example selection.


**Output Parsers**

Language models (and Chat Models) output text.
But many times you may want to get more structured information than just text back.
This is where output parsers come in.
Output Parsers are responsible for (1) instructing the model how output should be formatted,
(2) parsing output into the desired formatting (including retrying if necessary).


Go Deeper
---------

.. toctree::
   :maxdepth: 1

   ./prompts/prompt_templates.rst
   ./prompts/chat_prompt_template.ipynb
   ./prompts/example_selectors.rst
   ./prompts/output_parsers.rst
