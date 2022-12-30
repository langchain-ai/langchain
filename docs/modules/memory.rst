Memory
==========================

By default, Chains and Agents are stateless,
meaning that they treat each incoming query independently.
In some applications (chatbots being a GREAT example) it is highly important
to remember previous interactions, both at a short term but also at a long term level.
The concept of “Memory” exists to do exactly that.

The following sections of documentation are provided:

- `Getting Started <memory/getting_started.ipynb>`_: An overview of how to get started with different types of memory.

- `How-To Guides <memory/how_to_guides.rst>`_: A collection of how-to guides. These highlight how to work with different types of memory, as well as how to customize memory.

- `Reference </reference/memory.rst>`_: API reference documentation for all Memory classes.

- `Conceptual Guide <memory/conceptual_guide.md>`_: A conceptual guide going over the various concepts related to memory.


.. toctree::
   :maxdepth: 1
   :caption: Memory
   :name: Memory

   memory/getting_started.ipynb
   memory/how_to_guides.rst
   memory/key_concepts.rst
   /reference/memory.rst
