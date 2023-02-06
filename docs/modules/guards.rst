Guards
==========================

Guards are a set of directives that can be applies to chains, agents, tools, and generally any function that outputs a string. Guards are used to prevent a llm reliant function from outputting text that violates some constraint. For example, a guard can be used to prevent a chain from outputting text that includes profanity or which is in the wrong language.

One of the main reasons to use a guard is for security. Guards offer some protection against things like prompt leaking or users attempting to make agents output racist or otherwise offensive content. Guards can be used for many other uses, though. For example, if your application is specific to a certain industry you may add a guard to prevent agents from outputting irrelevant content.

The following sections of documentation are provided:

- `Getting Started <./guards/getting_started.html>`_: An overview of different types of guards and how to use them.

.. - `Key Concepts <./llms/key_concepts.html>`_: A conceptual guide going over the various concepts related to guards.

.. TODO: Probably want to add how-to guides for sentiment model guards!
.. - `How-To Guides <./llms/how_to_guides.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our LLM class, as well as how to integrate with various LLM providers.

- `Reference <../reference/modules/guards.html>`_: API reference documentation for all Guard classes.


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:
   
   ./llms/getting_started.ipynb
   ./llms/key_concepts.md
   ./llms/how_to_guides.rst
   Reference<../reference/modules/llms.rst>
