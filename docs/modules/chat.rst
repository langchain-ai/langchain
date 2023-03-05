Chat
==========================

Chat models are a variation on language models.
While chat models use language models under the hood, the interface they expose is a bit different.
Rather than expose a "text in, text out" API, they expose an interface where "chat messages" are the inputs and outputs.

Chat model APIs are fairly new, so we are still figuring out the correct abstractions.

The following sections of documentation are provided:

- `Getting Started <./chat/getting_started.html>`_: An overview of the basics of chat models.

- `Key Concepts <./chat/key_concepts.html>`_: A conceptual guide going over the various concepts related to chat models.

- `How-To Guides <./chat/how_to_guides.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our chat model class, as well as how to integrate with various chat model providers.


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:
   
   ./chat/getting_started.ipynb
   ./chat/key_concepts.md
   ./chat/how_to_guides.rst
