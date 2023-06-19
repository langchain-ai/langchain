How-To Guides
=============

LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. 
This is useful for logging, `tracing <../../additional_resources/tracing.html>`_, `streaming <../models/llms/examples/streaming_llm.html>`_, and other tasks.

You can subscribe to these events by using the `callbacks` argument available throughout the API. This 
argument is list of handler objects, which are expected to implement one or more of the methods described 
below in more detail. There are two main callbacks mechanisms:

* *Constructor callbacks* will be used for all calls made on that object, and will be scoped to that object 
only, i.e. if you pass a handler to the `LLMChain` constructor, it will not be used by the model attached 
to that chain. 

* *Request callbacks* will be used for that specific request only, and all sub-requests that it contains 
(eg. a call to an `LLMChain` triggers a call to a Model, which uses the same handler passed through). These 
are explicitly passed through.

**Advanced:** When you create a custom chain you can easily set it up to use the same callback system as all the built-in chains. 
`_call`, `_generate`, `_run`, and equivalent async methods on Chains / LLMs / Chat Models / Agents / Tools now receive a 2nd 
argument called `run_manager` which is bound to that run, and contains the logging methods that can be used by that object 
(i.e. `on_llm_new_token`). This is useful when constructing a custom chain. See this guide for more information on how to 
`create custom chains and use callbacks inside them <../chains/generic/custom_chain.html>`_.

**Examples**

This section contains examples of how to use the LangChain Callbacks.

.. toctree::
   :maxdepth: 1
   :glob:

   ./examples/*
