Welcome to LangChain
==========================

Large language models (LLMs) are emerging as a transformative technology, enabling
developers to build applications that they previously could not.
But using these LLMs in isolation is often not enough to
create a truly powerful app - the real power comes when you are able to
combine them with other sources of computation or knowledge.

This library is aimed at assisting in the development of those types of applications.
It aims to create:

1. a comprehensive collection of pieces you would ever want to combine
2. a flexible interface for combining pieces into a single comprehensive "chain"
3. a schema for easily saving and sharing those chains

The documentation is structured into the following sections:


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started

   getting_started/installation.md
   getting_started/environment.md
   getting_started/llm.md
   getting_started/chains.md

Goes over a simple walk through and tutorial for getting started setting up a simple chain that generates a company name based on what the company makes.
Covers installation, environment set up, calling LLMs, and using prompts.
Start here if you haven't used LangChain before.


.. toctree::
   :maxdepth: 1
   :caption: How-To Examples
   :name: examples

   examples/demos.rst
   examples/integrations.rst
   examples/prompts.rst
   examples/model_laboratory.ipynb

More elaborate examples and walk-throughs of particular
integrations and use cases. This is the place to look if you have questions
about how to integrate certain pieces, or if you want to find examples of
common tasks or cool demos.


.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference

   installation.md
   modules/prompt
   modules/llms
   modules/embeddings
   modules/text_splitter
   modules/vectorstore
   modules/chains


Full API documentation. This is the place to look if you want to
see detailed information about the various classes, methods, and APIs.


.. toctree::
   :maxdepth: 1
   :caption: Resources
   :name: resources

   core_concepts.md
   glossary.md
   Discord <https://discord.gg/6adMQxSpJS>

Higher level, conceptual explanations of the LangChain components.
This is the place to go if you want to increase your high level understanding
of the problems LangChain is solving, and how we decided to go about do so.

