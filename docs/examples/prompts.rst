LLMs & Prompts
==============

The examples here all highlight how to work with LLMs and prompts.

**LLMs**

`LLM Functionality <prompts/llm_functionality.ipynb>`_: A walkthrough of all the functionality the standard LLM interface exposes.

`LLM Serialization <prompts/llm_serialization.ipynb>`_: A walkthrough of how to serialize LLMs to and from disk.

`Custom LLM <prompts/custom_llm.ipynb>`_: How to create and use a custom LLM class, in case you have an LLM not from one of the standard providers (including one that you host yourself).


**Prompts**

`Prompt Management <prompts/prompt_management.ipynb>`_: A walkthrough of all the functionality LangChain supports for working with prompts.

`Prompt Serialization <prompts/prompt_serialization.ipynb>`_: A walkthrough of how to serialize prompts to and from disk.

`Few Shot Examples <prompts/few_shot_examples.ipynb>`_: How to include examples in the prompt.

`Generate Examples <prompts/generate_examples.ipynb>`_: How to use existing examples to generate more examples.

`Custom Example Selector <prompts/custom_example_selector.ipynb>`_: How to create and use a custom ExampleSelector (the class responsible for choosing which examples to use in a prompt).

`Custom Prompt Template <prompts/custom_prompt_template.ipynb>`_: How to create and use a custom PromptTemplate, the logic that decides how input variables get formatted into a prompt.


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   prompts/*
