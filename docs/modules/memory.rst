Memory
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/memory>`_


By default, Chains and Agents are stateless,
meaning that they treat each incoming query independently (as are the underlying LLMs and chat models).
In some applications (chatbots being a GREAT example) it is highly important
to remember previous interactions, both at a short term but also at a long term level.
The **Memory** does exactly that.

LangChain provides memory components in two forms.
First, LangChain provides helper utilities for managing and manipulating previous chat messages.
These are designed to be modular and useful regardless of how they are used.
Secondly, LangChain provides easy ways to incorporate these utilities into chains.

|
- `Getting Started <./memory/getting_started.html>`_: An overview of different types of memory.

- `How-To Guides <./memory/how_to_guides.html>`_: A collection of how-to guides. These highlight different types of memory, as well as how to use memory in chains.



.. toctree::
   :maxdepth: 1
   :caption: Memory
   :name: Memory
   :hidden:

   ./memory/getting_started.html
   ./memory/how_to_guides.rst
