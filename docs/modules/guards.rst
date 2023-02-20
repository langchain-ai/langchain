Guards
==========================

Guards are one way you can work on aligning your applications to prevent unwanted output or abuse. Guards are a set of directives that can be applied to chains, agents, tools, user inputs, and generally any function that outputs a string. Guards are used to prevent a llm reliant function from outputting text that violates some constraint and for preventing a user from inputting text that violates some constraint. For example, a guard can be used to prevent a chain from outputting text that includes profanity or which is in the wrong language.

Guards offer some protection against security or profanity related things like prompt leaking or users attempting to make agents output racist or otherwise offensive content. Guards can also be used for many other things, though. For example, if your application is specific to a certain industry you may add a guard to prevent agents from outputting irrelevant content or to prevent users from submitting off-topic questions.


- `Getting Started <./guards/getting_started.html>`_: An overview of different types of guards and how to use them.

- `Key Concepts <./guards/key_concepts.html>`_: A conceptual guide going over the various concepts related to guards.

.. TODO: Probably want to add how-to guides for sentiment model guards!
- `How-To Guides <./llms/how_to_guides.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our LLM class, as well as how to integrate with various LLM providers.

- `Reference <../reference/modules/guards.html>`_: API reference documentation for all Guard classes.


.. toctree::
   :maxdepth: 1
   :name: Guards
   :hidden:
   
   ./guards/getting_started.ipynb
   ./guards/key_concepts.md
   Reference<../reference/modules/guards.rst>

