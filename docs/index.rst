Welcome to LangChain
==========================

| **LangChain** is a framework for developing applications powered by language models. It enables applications that are
1. **Data-aware**: connect a language model to other sources of data
2. **Agentic**: allow a language model to interact with its environment

| Note: These docs are Python-specific. There are `separate docs for JavaScript <https://js.langchain.com/docs/>`_.

Get started
----------------

We recommend heading to our `Quickstart Guide <./getting_started/getting_started.html>`_ to get set up and to familiarize yourself with key concepts.

.. toctree::
   :maxdepth: 2
   :caption: Get started
   :name: getting_started
   :hidden:

   getting_started/getting_started.html
   getting_started/installation.html

Modules
-----------

| LangChain provides standard, extendable interfaces and external integrations for the following modules (from least to most complex):

- `Model I/O <./modules/model_io.html>`_: Interface with language models

- `Data I/O <./modules/data_io.html>`_: Interface with application-specific data

- `Chains <./modules/chains.html>`_: Structured sequences of calls

- `Memory <./modules/memory.html>`_: Application state that is persisted between calls of a chain

- `Agents <./modules/agents.html>`_: Chains that choose how to execute high-level directives given a set of tools

- `Callbacks <./modules/callbacks/getting_started.html>`_: Logging and streaming of intermediate steps of any chain

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/model_io.html
   ./modules/data_io.html
   ./modules/chains.rst
   ./modules/memory.rst
   ./modules/agents.rst
   ./modules/callbacks/getting_started.ipynb

Examples
------------

| Walkthroughs for common use cases and additional examples of applications built on LangChain:

- `Use cases <./use_cases/autonomous_agents.html>`_

- `Additional resources <./additional_resources/youtube.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :name: examples
   :hidden:

   ./use_cases.rst
   ./additional_resources.rst


Ecosystem
------------

| LangChain is part of a rich ecosystem of tools integrated with the framework and applications built on top of it:

- `Integrations <./integrations.html>`_: Use your favorite models, tools and applications within LangChain

- `Dependents <./dependents.html>`_: Repositories that use LangChain

- `Deployments <./ecosystem/deployments.html>`_: Instructions, code snippets, and template repositories for deploying LangChain apps


.. toctree::
   :maxdepth: 1
   :caption: Ecosystem
   :name: ecosystem
   :hidden:

   ./integrations.rst
   ./ecosystem/deployments.md
   LangChainHub <https://github.com/hwchase17/langchain-hub>
   ./dependents.md


.. toctree::
   :maxdepth: 1
   :caption: Find us
   :name: find_us
   :hidden:

   Twitter <https://twitter.com/LangChainAI>
   Discord <https://discord.gg/6adMQxSpJS>
   Request production support <https://forms.gle/57d8AmXBYp8PP8tZA>
