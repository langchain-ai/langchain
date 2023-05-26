Welcome to LangChain
==========================

Overview
----------------

| **LangChain** is a framework for developing applications powered by language models. It enables applications that are
1. **Data-aware**: connect a language model to other sources of data
2. **Agentic**: allow a language model to interact with its environment

| Note: this documentation is Python-specific. There is a separate `Conceptual Guide <https://docs.langchain.com/docs/>`_ and `JavaScript documentation <https://js.langchain.com/docs/>`_.

Getting Started
----------------

We recommend heading to our `Quickstart Guide <./getting_started/getting_started.html>`_ to get set up and to familiarize yourself with key concepts.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started/getting_started.md
   getting_started/concepts.md
   getting_started/tutorials.md

Modules
-----------

| LangChain provides standard, extendable interfaces and external integrations for the following modules (from least to most complex)

- `Models <./modules/models.html>`_: LLMs, chat models, text embedding models

- `Prompts <./modules/prompts.html>`_: Prompt management, optimization, and serialization

- `Memory <./modules/memory.html>`_: State that is persisted between calls of a chain/agent

- `Indexes <./modules/indexes.html>`_: Connect language models to application-specific data

- `Chains <./modules/chains.html>`_: Structured sequences of calls

- `Agents <./modules/agents.html>`_: LLMs that execute high-level directives given a set of tools

- `Callbacks <./modules/callbacks/getting_started.html>`_: Log and stream intermediate steps of any chain

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/models.rst
   ./modules/prompts.rst
   ./modules/memory.rst
   ./modules/indexes.rst
   ./modules/chains.rst
   ./modules/agents.rst
   ./modules/callbacks/getting_started.ipynb

Use Cases
----------

| Best practices and built-in implementations for common use cases

- `Autonomous Agents <./use_cases/autonomous_agents.html>`_: Long-running agents that take many steps, like AutoGPT and BabyAGI

- `Personal Assistants <./use_cases/personal_assistants.html>`_: Taking actions, storing interactions, and connecting to data

- `Question Answering <./use_cases/question_answering.html>`_: Answering questions over specific documents

- `Chatbots <./use_cases/chatbots.html>`_: Long-running conversations

- `Data Analysis <./use_cases/tabular.html>`_: Using language models to query structured data

- `Code Understanding <./use_cases/code.html>`_: Using language models to analyze code

- `Interacting with APIs <./use_cases/apis.html>`_: Enabling language models to interact with APIs

- `Information Extraction <./use_cases/extraction.html>`_: Extract structured information from text

- `Summarization <./use_cases/summarization.html>`_: Compressing long text

- `Evaluation <./use_cases/evaluation.html>`_: Using language models to evaluate language models

- `Agent Simulations <./use_cases/agent_simulations.html>`_: Putting agents in a sandbox and observing them


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/autonomous_agents.md
   ./use_cases/agent_simulations.md
   ./use_cases/personal_assistants.md
   ./use_cases/question_answering.md
   ./use_cases/chatbots.md
   ./use_cases/tabular.rst
   ./use_cases/code.md
   ./use_cases/apis.md
   ./use_cases/extraction.md
   ./use_cases/summarization.md
   ./use_cases/evaluation.rst


Reference
---------------

| All methods, classes, installation methods, and integration setups

- `Installation <./reference/installation.html>`_

- `API Reference <./reference.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference
   :hidden:

   ./reference/installation.md
   ./reference.rst


Ecosystem
------------

| LangChain has integrations for many models, tools and applications, and many applications are built using LangChain

- `Integrations <./integrations.html>`_: Use your favorite models, tools and applications within LangChain

- `Dependents <./dependents.html>`_: Repositories that use LangChain

- `Deployments <./ecosystem/deployments.html>`_: Instructions, code snippets, and template repositories for deploying LangChain apps


.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Ecosystem
   :name: ecosystem
   :hidden:

   ./integrations.rst
   ./dependents.md
   ./ecosystem/deployments.md


Additional Resources
---------------------

- `LangChainHub <https://github.com/hwchase17/langchain-hub>`_: Share and explore other prompts, chains, and agents

- `Gallery <https://github.com/kyrolabs/awesome-langchain>`_: Great projects that use Langchain, compiled by the folks at `Kyrolabs <https://kyrolabs.com>`_

- `Tracing <./additional_resources/tracing.html>`_: Log and visualize the execution of chains and agents

- `Model Laboratory <./additional_resources/model_laboratory.html>`_: Experimenting with different prompts, models, and chains

- `YouTube <./additional_resources/youtube.html>`_: Video tutorials

- `Discord <https://discord.gg/6adMQxSpJS>`_: Discuss and share all things LangChain!

- `Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>`_: Get a dedicated Slack channel with the LangChain team as you move your applications into production


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :name: resources
   :hidden:

   LangChainHub <https://github.com/hwchase17/langchain-hub>
   Gallery <https://github.com/kyrolabs/awesome-langchain>
   ./additional_resources/tracing.md
   ./additional_resources/model_laboratory.ipynb
   ./additional_resources/youtube.md
   Discord <https://discord.gg/6adMQxSpJS>
   Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>
