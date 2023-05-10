Models
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/models>`_


This section of the documentation deals with different types of models that are used in LangChain.
On this page we will go over the model types at a high level,
but we have individual pages for each model type.
The pages contain more detailed "how-to" guides for working with that model,
as well as a list of different model providers.

**LLMs**

Large Language Models (LLMs) are the first type of models we cover.
These models take a text string as input, and return a text string as output.


**Chat Models**

Chat Models are the second type of models we cover.
These models are usually backed by a language model, but their APIs are more structured.
Specifically, these models take a list of Chat Messages as input, and return a Chat Message.

**Text Embedding Models**

The third type of models we cover are text embedding models.
These models take text as input and return a list of floats.


Go Deeper
---------

.. toctree::
   :maxdepth: 1

   ./models/llms.rst
   ./models/chat.rst
   ./models/text_embedding.rst
