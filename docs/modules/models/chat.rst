Chat Models
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/models/chat-model>`_


Chat models are a variation on language models.
While chat models use language models under the hood, the interface they expose is a bit different.
Rather than expose a "text in, text out" API, they expose an interface where "chat messages" are the inputs and outputs.

Chat model APIs are fairly new, so we are still figuring out the correct abstractions.

The following sections of documentation are provided:

- `Getting Started <./chat/getting_started.html>`_: An overview of all the functionality the LangChain LLM class provides.

- `How-To Guides <./chat/how_to_guides.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our LLM class (streaming, async, etc).

- `Integrations <./chat/integrations.html>`_: A collection of examples on how to integrate different LLM providers with LangChain (OpenAI, Hugging Face, etc).


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:

   ./chat/getting_started.ipynb
   ./chat/how_to_guides.rst
   ./chat/integrations.rst
