Welcome to LangChain
==========================

Large language models (LLMs) are emerging as a transformative technology, enabling
developers to build applications that they previously could not.
But using these LLMs in isolation is often not enough to
create a truly powerful app - the real power comes when you are able to
combine them with other sources of computation or knowledge.

This library is aimed at assisting in the development of those types of applications.

Getting Started
----------------

To get started with LangChain, including installation and environment setup, checkout the below guide.

- `Getting Started Documentation <getting_started>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started.rst

Modules
-----------

There are six main modules that LangChain provides support for.
For each module we provide some examples to get started, how-to guides, reference docs, and conceptual guides.
These modules are, in increasing order of complexity:


- `Prompts <modules/prompts>`_: This includes prompt management, prompt optimization, and prompt serialization.

- `LLMs <modules/llms>`_: This includes a generic interface for all LLMs, and common utilities for working with LLMs.

- `Utils <modules/utils>`_: Language models are often more powerful when interacting with other sources of knowledge or computation. This can include Python REPLs, embeddings, search engines, and more. LangChain provides a large collection of common utils to use in your application.

- `Chains <modules/chains>`_: Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

- `Agents <modules/agents>`_: Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.

- `Memory <modules/memory>`_: Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.


.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   modules/prompts.md
   modules/llms.md
   modules/utils.md
   modules/chains.md
   modules/agents.md
   modules/memory.md

Use Cases
----------

The above modules can be used in a variety of ways. LangChain also provides guidance and assistance in this. Below are some of the common use cases LangChain supports.

`Data Augmented Generation <use_cases/combine_docs>`_: Data Augmented Generation involves specific types of chains that first interact with an external datasource to fetch data to use in the generation step. Examples of this include summarization of long pieces of text and question/answering over specific data sources.

`QA with Sources <use_cases/qa_with_sources>`_: Answering questions over specific documents, while also making sure to say what source it got its information from. A type of Data Augmented Generation.

`Question Answering <use_cases/question_answering>`_: Answering questions over specific documents, only utilizing the information in those documents to construct an answer. A type of Data Augmented Generation.

`Summarization <use_cases/summarization>`_: Summarizing longer documents into shorter, more condensed chunks of information. A type of Data Augmented Generation.

`Evaluation <use_cases/evaluation>`_: Generative models are notoriously hard to evaluate with traditional metrics. One new way of evaluating them is using language models themselves to do the evaluation. LangChain provides some prompts/chains for assisting in this.

`Model Laboratory <use_cases/model_laboratory>`_: Experimenting with different prompts, models, and chains is a big part of developing the best possible application. The ModelLaboratory makes it easy to do so.




.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   use_cases/combine_docs.md
   use_cases/qa_with_sources.md
   use_cases/question_answering.md
   use_cases/summarization.md
   use_cases/evaluation.rst
   use_cases/model_laboratory.ipynb


Reference Docs
---------------

All of LangChain's reference documentation, in one place.

- `Reference Documentation <reference>`_: Full documentation on all methods, classes, and installation methods for LangChain.

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference
   :hidden:

   reference.rst


Additional Resources
---------------------

Additional collection of resources we think may be useful as you develop your application!


- `Glossary <glossary>`_: A glossary of all related terms, papers, methods, etc. Whether implemented in LangChain or not!

- `Gallery <gallery>`_: A collection of our favorite projects that use LangChain. Useful for finding inspiration or seeing how things were done in other applications.

- `Discord <https://discord.gg/6adMQxSpJS>`_: Join us on our Discord to discuss all things LangChain!


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :name: resources
   :hidden:

   glossary.md
   gallery.md

