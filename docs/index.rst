Welcome to LangChain
==========================

Large language models (LLMs) are emerging as a transformative technology, enabling
developers to build applications that they previously could not.
But using these LLMs in isolation is often not enough to
create a truly powerful app - the real power comes when you are able to
combine them with other sources of computation or knowledge.

This library is aimed at assisting in the development of those types of applications. Common examples of these types of applications include:

**❓ Question Answering over specific documents**

- `Documentation <./use_cases/question_answering.html>`_
- End-to-end Example: `Question Answering over Notion Database <https://github.com/hwchase17/notion-qa>`_

**💬 Chatbots**

- `Documentation <./use_cases/chatbots.html>`_
- End-to-end Example: `Chat-LangChain <https://github.com/hwchase17/chat-langchain>`_

**🤖 Agents**

- `Documentation <./use_cases/agents.html>`_
- End-to-end Example: `GPT+WolframAlpha <https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain>`_

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


- `Prompts <./modules/prompts.html>`_: This includes prompt management, prompt optimization, and prompt serialization.

- `LLMs <./modules/llms.html>`_: This includes a generic interface for all LLMs, and common utilities for working with LLMs.

- `Document Loaders <./modules/document_loaders.html>`_: This includes a standard interface for loading documents, as well as specific integrations to all types of text data sources.

- `Utils <./modules/utils.html>`_: Language models are often more powerful when interacting with other sources of knowledge or computation. This can include Python REPLs, embeddings, search engines, and more. LangChain provides a large collection of common utils to use in your application.

- `Chains <./modules/chains.html>`_: Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

- `Indexes <./modules/indexes.html>`_: Language models are often more powerful when combined with your own text data - this module covers best practices for doing exactly that.

- `Agents <./modules/agents.html>`_: Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.

- `Memory <./modules/memory.html>`_: Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.


.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/prompts.md
   ./modules/llms.md
   ./modules/document_loaders.md
   ./modules/utils.md
   ./modules/indexes.md
   ./modules/chains.md
   ./modules/agents.md
   ./modules/memory.md

Use Cases
----------

The above modules can be used in a variety of ways. LangChain also provides guidance and assistance in this. Below are some of the common use cases LangChain supports.

- `Agents <./use_cases/agents.html>`_: Agents are systems that use a language model to interact with other tools. These can be used to do more grounded question/answering, interact with APIs, or even take actions.

- `Chatbots <./use_cases/chatbots.html>`_: Since language models are good at producing text, that makes them ideal for creating chatbots.

- `Data Augmented Generation <./use_cases/combine_docs.html>`_: Data Augmented Generation involves specific types of chains that first interact with an external datasource to fetch data to use in the generation step. Examples of this include summarization of long pieces of text and question/answering over specific data sources.

- `Question Answering <./use_cases/question_answering.html>`_: Answering questions over specific documents, only utilizing the information in those documents to construct an answer. A type of Data Augmented Generation.

- `Summarization <./use_cases/summarization.html>`_: Summarizing longer documents into shorter, more condensed chunks of information. A type of Data Augmented Generation.

- `Evaluation <./use_cases/evaluation.html>`_: Generative models are notoriously hard to evaluate with traditional metrics. One new way of evaluating them is using language models themselves to do the evaluation. LangChain provides some prompts/chains for assisting in this.

- `Generate similar examples <./use_cases/generate_examples.html>`_: Generating similar examples to a given input. This is a common use case for many applications, and LangChain provides some prompts/chains for assisting in this.

- `Compare models <./use_cases/model_laboratory.html>`_: Experimenting with different prompts, models, and chains is a big part of developing the best possible application. The ModelLaboratory makes it easy to do so.



.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/agents.md
   ./use_cases/chatbots.md
   ./use_cases/generate_examples.ipynb
   ./use_cases/combine_docs.md
   ./use_cases/question_answering.md
   ./use_cases/summarization.md
   ./use_cases/evaluation.rst
   ./use_cases/model_laboratory.ipynb


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

- `Discord <https://discord.gg/6adMQxSpJS>`_: Join us on our Discord to discuss all things LangChain!

- `Tracing <./tracing.html>`_: A guide on using tracing in LangChain to visualize the execution of chains and agents.

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
   Discord <https://discord.gg/6adMQxSpJS>
   Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>
