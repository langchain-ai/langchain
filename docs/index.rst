Welcome to LangChain
==========================

| **LangChain** is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model, but will also be:
1. *Data-aware*: connect a language model to other sources of data
2. *Agentic*: allow a language model to interact with its environment

| The LangChain framework is designed around these principles.

| This is the Python specific portion of the documentation. For a purely conceptual guide to LangChain, see `here <https://docs.langchain.com/docs/>`_. For the JavaScript documentation, see `here <https://js.langchain.com/docs/>`_.

Getting Started
----------------

| How to get started using LangChain to create an Language Model application.

- `Quickstart Guide <./getting_started/getting_started.html>`_

| Concepts and terminology.

- `Concepts and terminology <./getting_started/concepts.html>`_

| Tutorials created by community experts and presented on YouTube.

- `Tutorials <./getting_started/tutorials.html>`_

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

| These modules are the core abstractions which we view as the building blocks of any LLM-powered application.
For each module LangChain provides standard, extendable interfaces. LangChain also provides external integrations and even end-to-end implementations for off-the-shelf use.

| The docs for each module contain quickstart examples, how-to guides, reference docs, and conceptual guides.

| The modules are (from least to most complex):

- `Models <./modules/models.html>`_: Supported model types and integrations.

- `Prompts <./modules/prompts.html>`_: Prompt management, optimization, and serialization.

- `Memory <./modules/memory.html>`_: Memory refers to state that is persisted between calls of a chain/agent.

- `Indexes <./modules/indexes.html>`_: Language models become much more powerful when combined with application-specific data - this module contains interfaces and integrations for loading, querying and updating external data.

- `Chains <./modules/chains.html>`_: Chains are structured sequences of calls (to an LLM or to a different utility).

- `Agents <./modules/agents.html>`_: An agent is a Chain in which an LLM, given a high-level directive and a set of tools, repeatedly decides an action, executes the action and observes the outcome until the high-level directive is complete.

- `Callbacks <./modules/callbacks/getting_started.html>`_: Callbacks let you log and stream the intermediate steps of any chain, making it easy to observe, debug, and evaluate the internals of an application.

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/models.rst
   ./modules/prompts.rst
   ./modules/memory.md
   ./modules/indexes.md
   ./modules/chains.md
   ./modules/agents.md
   ./modules/callbacks/getting_started.ipynb

Use Cases
----------

| Best practices and built-in implementations for common LangChain use cases:

- `Autonomous Agents <./use_cases/autonomous_agents.html>`_: Autonomous agents are long-running agents that take many steps in an attempt to accomplish an objective. Examples include AutoGPT and BabyAGI.

- `Agent Simulations <./use_cases/agent_simulations.html>`_: Putting agents in a sandbox and observing how they interact with each other and react to events can be an effective way to evaluate their long-range reasoning and planning abilities.

- `Personal Assistants <./use_cases/personal_assistants.html>`_: One of the primary LangChain use cases. Personal assistants need to take actions, remember interactions, and have knowledge about your data.

- `Question Answering <./use_cases/question_answering.html>`_: Another common LangChain use case. Answering questions over specific documents, only utilizing the information in those documents to construct an answer.

- `Chatbots <./use_cases/chatbots.html>`_: Language models love to chat, making this a very natural use of them.

- `Querying Tabular Data <./use_cases/tabular.html>`_: Recommended reading if you want to use language models to query structured data (CSVs, SQL, dataframes, etc).

- `Code Understanding <./use_cases/code.html>`_: Recommended reading if you want to use language models to analyze code.

- `Interacting with APIs <./use_cases/apis.html>`_: Enabling language models to interact with APIs is extremely powerful. It gives them access to up-to-date information and allows them to take actions.

- `Extraction <./use_cases/extraction.html>`_: Extract structured information from text.

- `Summarization <./use_cases/summarization.html>`_: Compressing longer documents. A type of Data-Augmented Generation.

- `Evaluation <./use_cases/evaluation.html>`_: Generative models are hard to evaluate with traditional metrics. One promising approach is to use language models themselves to do the evaluation.


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


Reference Docs
---------------

| Full documentation on all methods, classes, installation methods, and integration setups for LangChain.


- `LangChain Installation <./reference/installation.html>`_

- `Reference Documentation <./reference.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference
   :hidden:

   ./reference/installation.md
   ./reference.rst


Ecosystem
------------

| LangChain integrates a lot of different LLMs, systems, and products.
| From the other side, many systems and products depend on LangChain.
| It creates a vibrant and thriving ecosystem.


- `Integrations <./integrations.html>`_: Guides for how other products can be used with LangChain.

- `Dependents <./dependents.html>`_: List of repositories that use LangChain.

- `Deployments <./ecosystem/deployments.html>`_: A collection of instructions, code snippets, and template repositories for deploying LangChain apps.


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

| Additional resources we think may be useful as you develop your application!

- `LangChainHub <https://github.com/hwchase17/langchain-hub>`_: The LangChainHub is a place to share and explore other prompts, chains, and agents.

- `Gallery <https://github.com/kyrolabs/awesome-langchain>`_: A collection of great projects that use Langchain, compiled by the folks at `Kyrolabs <https://kyrolabs.com>`_. Useful for finding inspiration and example implementations.

- `Tracing <./additional_resources/tracing.html>`_: A guide on using tracing in LangChain to visualize the execution of chains and agents.

- `Model Laboratory <./additional_resources/model_laboratory.html>`_: Experimenting with different prompts, models, and chains is a big part of developing the best possible application. The ModelLaboratory makes it easy to do so.

- `Discord <https://discord.gg/6adMQxSpJS>`_: Join us on our Discord to discuss all things LangChain!

- `YouTube <./additional_resources/youtube.html>`_: A collection of the LangChain tutorials and videos.

- `Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>`_: As you move your LangChains into production, we'd love to offer more comprehensive support. Please fill out this form and we'll set up a dedicated support Slack channel.


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :name: resources
   :hidden:

   LangChainHub <https://github.com/hwchase17/langchain-hub>
   Gallery <https://github.com/kyrolabs/awesome-langchain>
   ./additional_resources/tracing.md
   ./additional_resources/model_laboratory.ipynb
   Discord <https://discord.gg/6adMQxSpJS>
   ./additional_resources/youtube.md
   Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>
