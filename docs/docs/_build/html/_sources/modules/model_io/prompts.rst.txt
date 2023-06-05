Prompts
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/prompts>`_


The new way of programming models is through prompts.
A **prompt** refers to the input to the model.
This input is often constructed from multiple components.
A **PromptTemplate** is responsible for the construction of this input.
LangChain provides several classes and functions to make constructing and working with prompts easy.

|
- `Getting Started <./prompts/getting_started.html>`_: An overview of the prompts.


- `LLM Prompt Templates <./prompts/prompt_templates.html>`_: How to use PromptTemplates to prompt Language Models.


- `Chat Prompt Templates <./prompts/chat_prompt_template.html>`_: How to use PromptTemplates to prompt Chat Models.


- `Example Selectors <./prompts/example_selectors.html>`_: Often times it is useful to include examples in prompts.
  These examples can be dynamically selected. This section goes over example selection.


- `Output Parsers <./prompts/output_parsers.html>`_: Language models (and Chat Models) output text.
  But many times you may want to get more structured information. This is where output parsers come in.
  Output Parsers:

  - instruct the model how output should be formatted,
  - parse output into the desired formatting (including retrying if necessary).



.. toctree::
   :maxdepth: 1
   :caption: Prompts
   :name: prompts
   :hidden:

   ./prompts/getting_started.html
   ./prompts/prompt_templates.rst
   ./prompts/chat_prompt_template.html
   ./prompts/example_selectors.rst
   ./prompts/output_parsers.rst
