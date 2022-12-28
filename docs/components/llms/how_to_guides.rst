How-To Guides
=============

The examples here all highlight how to work with LLMs.


**LLMs**

`LLM Serialization <examples/llm_serialization.ipynb>`_: A walkthrough of how to serialize LLMs to and from disk.

`LLM Caching <examples/llm_caching.ipynb>`_: Covers different types of caches, and how to use a cache to save results of LLM calls.

`Custom LLM <examples/custom_llm.ipynb>`_: How to create and use a custom LLM class, in case you have an LLM not from one of the standard providers (including one that you host yourself).


**Specific LLM Integrations**

`Huggingface Hub <examples/huggingface_hub.ipynb>`_: Covers how to connect to LLMs hosted on HuggingFace Hub.

`Azure OpenAI <examples/azure_openai_example.ipynb>`_: Covers how to connect to Azure-hosted OpenAI Models.


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   examples/*