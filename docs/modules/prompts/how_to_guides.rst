How-To Guides
=============

If you're new to the library, you may want to start with the `Quickstart <getting_started.md>`_.

The user guide here shows more advanced workflows and how to use the library in different ways.

`Prompt Serialization <examples/prompt_serialization.ipynb>`_: A walkthrough of how to serialize prompts to and from disk.

`Few Shot Examples <examples/few_shot_examples.ipynb>`_: How to include examples in the prompt.

`Generate Examples <examples/generate_examples.ipynb>`_: How to use existing examples to generate more examples.

`Custom Example Selector <examples/custom_example_selector.ipynb>`_: How to create and use a custom ExampleSelector (the class responsible for choosing which examples to use in a prompt).

`Custom Prompt Template <examples/custom_prompt_template.md>`_: How to create and use a custom PromptTemplate, the logic that decides how input variables get formatted into a prompt.


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   examples/*