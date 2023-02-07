Alignment
==========================

Alignment in the context of LLMs is the process of steering a LLM towards a desired behavior or outcome. Alignment tools and methods can be subdivided into three broad categories based on where in the process of user interaction to LLM response they fall:

- Methods that aim to modify or block user input to ensure an aligned response
- Methods that aim to modify the LLM itself to ensure an aligned response
- Methods that aim to modify or block the LLM response to ensure an aligned response

Currently, aligning the LLM itself (ensuring that the LLM always gives safe, accurate, and desired responses when prompted) is an open area of research. LangChain provides tools to assist in alignment via the other two methods, limiting user input and limiting LLM responses.

---------------------
Guards
---------------------

Guards are one way you can work on aligning your applications to prevent unwanted output or abuse. Guards are a set of directives that can be applied to chains, agents, tools, user inputs, and generally any function that outputs a string. Guards are used to prevent a llm reliant function from outputting text that violates some constraint and for preventing a user from inputting text that violates some constraint. For example, a guard can be used to prevent a chain from outputting text that includes profanity or which is in the wrong language.

One of the main goals of alignment is safety and security. Guards offer some protection in these areas against things like prompt leaking or users attempting to make agents output racist or otherwise offensive content. Guards can also be used for many other things, though. For example, if your application is specific to a certain industry you may add a guard to prevent agents from outputting irrelevant content.


- `Getting Started <./alignment/guards/getting_started.html>`_: An overview of different types of guards and how to use them.

- `Key Concepts <./alignment/guards/key_concepts.html>`_: A conceptual guide going over the various concepts related to guards.

.. TODO: Probably want to add how-to guides for sentiment model guards!
.. - `How-To Guides <./alignment/llms/how_to_guides.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our LLM class, as well as how to integrate with various LLM providers.

- `Reference <../reference/modules/guards.html>`_: API reference documentation for all Guard classes.


.. toctree::
   :maxdepth: 1
   :name: Guards
   :hidden:
   
   ./alignment/guards/getting_started.ipynb
   ./alignment/guards/key_concepts.md
   Reference<../reference/modules/guards.rst>

