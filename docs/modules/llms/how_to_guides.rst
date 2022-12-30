How-To Guides
=============

The examples here all address certain "how-to" guides for working with LLMs.


**Generic Functionality**

`LLM Serialization <examples/llm_serialization>`_: A walkthrough of how to serialize LLMs to and from disk.

`LLM Caching <examples/llm_caching>`_: Covers different types of caches, and how to use a cache to save results of LLM calls.

`Custom LLM <examples/custom_llm>`_: How to create and use a custom LLM class, in case you have an LLM not from one of the standard providers (including one that you host yourself).


**Specific LLM Integrations**

`Huggingface Hub <examples/huggingface_hub>`_: Covers how to connect to LLMs hosted on HuggingFace Hub.

`Azure OpenAI <examples/azure_openai_example>`_: Covers how to connect to Azure-hosted OpenAI Models.

`Manifest <examples/manifest>`_: Covers how to utilize the Manifest wrapper.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Generic Functionality
   :name: Generic Functionality
   :hidden:

   examples/*

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Specific LLM Integrations
   :name: Specific LLM Integrations
   :hidden:

   integrations/*