How-To Guides
=============

A chain is made up of links, which can be either primitives or other chains.
Primitives can be either `prompts <../prompts.html>`_, `llms <../llms.html>`_, `utils <../utils.html>`_, or other chains.
The examples here are all end-to-end chains for specific applications.
They are broken up into three categories:

1. `Generic Chains <./generic_how_to.html>`_: Generic chains, that are meant to help build other chains rather than serve a particular purpose.
2. `CombineDocuments Chains <./combine_docs_how_to.html>`_: Chains aimed at making it easy to work with documents (question answering, summarization, etc).
3. `Utility Chains <./utility_how_to.html>`_: Chains consisting of an LLMChain interacting with a specific util.

.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   ./generic_how_to.rst
   ./combine_docs_how_to.rst
   ./utility_how_to.rst
