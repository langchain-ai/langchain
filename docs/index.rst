Welcome to LangChain
==========================

LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an API, but will also:

- *Be data-aware*: connect a language model to other sources of data
- *Be agentic*: allow a language model to interact with its environment

The LangChain framework is designed with the above principles in mind.

This is the Python specific portion of the documentation. For a purely conceptual guide to LangChain, see `here <https://docs.langchain.com/docs/>`_. For the JavaScript documentation, see `here <https://js.langchain.com/docs/>`_.

Getting Started
----------------

Checkout the below guide for a walkthrough of how to get started using LangChain to create an Language Model application.

- `Getting Started Documentation <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started/getting_started.md

Modules
-----------

There are several main modules that LangChain provides support for.
For each module we provide some examples to get started, how-to guides, reference docs, and conceptual guides.
These modules are, in increasing order of complexity:

- `Models <./modules/models.html>`_: The various model types and model integrations LangChain supports.

- `Prompts <./modules/prompts.html>`_: This includes prompt management, prompt optimization, and prompt serialization.

- `Memory <./modules/memory.html>`_: Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.

- `Indexes <./modules/indexes.html>`_: Language models are often more powerful when combined with your own text data - this module covers best practices for doing exactly that.

- `Chains <./modules/chains.html>`_: Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

- `Agents <./modules/agents.html>`_: Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.


.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/models.rst
   ./modules/prompts.rst
   ./modules/indexes.md
   ./modules/memory.md
   ./modules/chains.md
   ./modules/agents.md

Use Cases
----------

The above modules can be used in a variety of ways. LangChain also provides guidance and assistance in this. Below are some of the common use cases LangChain supports.

- `Autonomous Agents <./use_cases/autonomous_agents.html>`_: Autonomous agents are long running agents that take many steps in an attempt to accomplish an objective. Examples include AutoGPT and BabyAGI.

- `Agent Simulations <./use_cases/agent_simulations.html>`_: Putting agents in a sandbox and observing how they interact with each other or to events can be an interesting way to observe their long-term memory abilities.

- `Personal Assistants <./use_cases/personal_assistants.html>`_: The main LangChain use case. Personal assistants need to take actions, remember interactions, and have knowledge about your data.

- `Question Answering <./use_cases/question_answering.html>`_: The second big LangChain use case. Answering questions over specific documents, only utilizing the information in those documents to construct an answer.

- `Chatbots <./use_cases/chatbots.html>`_: Since language models are good at producing text, that makes them ideal for creating chatbots.

- `Querying Tabular Data <./use_cases/tabular.html>`_: If you want to understand how to use LLMs to query data that is stored in a tabular format (csvs, SQL, dataframes, etc) you should read this page.

- `Code Understanding <./use_cases/code.html>`_: If you want to understand how to use LLMs to query source code from github, you should read this page.

- `Interacting with APIs <./use_cases/apis.html>`_: Enabling LLMs to interact with APIs is extremely powerful in order to give them more up-to-date information and allow them to take actions.

- `Extraction <./use_cases/extraction.html>`_: Extract structured information from text.

- `Summarization <./use_cases/summarization.html>`_: Summarizing longer documents into shorter, more condensed chunks of information. A type of Data Augmented Generation.

- `Evaluation <./use_cases/evaluation.html>`_: Generative models are notoriously hard to evaluate with traditional metrics. One new way of evaluating them is using language models themselves to do the evaluation. LangChain provides some prompts/chains for assisting in this.


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/personal_assistants.md
   ./use_cases/autonomous_agents.md
   ./use_cases/agent_simulations.md
   ./use_cases/question_answering.md
   ./use_cases/chatbots.md
   ./use_cases/tabular.rst
   ./use_cases/code.md
   ./use_cases/apis.md
   ./use_cases/summarization.md
   ./use_cases/extraction.md
   ./use_cases/evaluation.rst


Reference Docs
---------------

All of LangChain's reference documentation, in one place. Full documentation on all methods, classes, installation methods, and integration setups for LangChain.


- `Reference Documentation <./reference.html>`_
.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference
   :hidden:

   ./reference/installation.md
   ./reference/integrations.md
   ./reference.rst


LangChain Ecosystem
-------------------

Guides for how other companies/products can be used with LangChain

- `LangChain Ecosystem <./ecosystem.html>`_

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Ecosystem
   :name: ecosystem
   :hidden:

   ./ecosystem.rst


Additional Resources
---------------------

Additional collection of resources we think may be useful as you develop your application!

- `LangChainHub <https://github.com/hwchase17/langchain-hub>`_: The LangChainHub is a place to share and explore other prompts, chains, and agents.

- `Glossary <./glossary.html>`_: A glossary of all related terms, papers, methods, etc. Whether implemented in LangChain or not!

- `Gallery <./gallery.html>`_: A collection of our favorite projects that use LangChain. Useful for finding inspiration or seeing how things were done in other applications.

- `Deployments <./deployments.html>`_: A collection of instructions, code snippets, and template repositories for deploying LangChain apps.

- `Tracing <./tracing.html>`_: A guide on using tracing in LangChain to visualize the execution of chains and agents.

- `Model Laboratory <./model_laboratory.html>`_: Experimenting with different prompts, models, and chains is a big part of developing the best possible application. The ModelLaboratory makes it easy to do so.

- `Discord <https://discord.gg/6adMQxSpJS>`_: Join us on our Discord to discuss all things LangChain!

- `YouTube <./youtube.html>`_: A collection of the LangChain tutorials and videos.

- `Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>`_: As you move your LangChains into production, we'd love to offer more comprehensive support. Please fill out this form and we'll set up a dedicated support Slack channel.


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :name: resources
   :hidden:

   LangChainHub <https://github.com/hwchase17/langchain-hub>
   ./glossary.md
   ./gallery.rst
   ./deployments.md
   ./tracing.md
   ./use_cases/model_laboratory.ipynb
   Discord <https://discord.gg/6adMQxSpJS>
   ./youtube.md
   Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>
