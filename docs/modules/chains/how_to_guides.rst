How-To Guides
=============

A chain is made up of links, which can be either primitives or other chains.
Primitives can be either `prompts <../prompts.html>`_, `llms <../llms.html>`_, `utils <../utils.html>`_, or other chains.
The examples here are all end-to-end chains for specific applications.
They are broken up into two categories:

1. Generic Chains
2. CombineDocuments Chains

Generic Chains
--------------

**LLMChain**

- **Links Used**: PromptTemplate, LLM
- **Notes**: This chain is the simplest chain, and is widely used by almost every other chain. This chain takes arbitrary user input, creates a prompt with it from the PromptTemplate, passes that to the LLM, and then returns the output of the LLM as the final output.
- `Example Notebook <examples/llm_chain.html>`_

**LLMMath**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a math question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Example Notebook <examples/llm_math.html>`_

**PAL**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a reasoning question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Paper <https://arxiv.org/abs/2211.10435>`_
- `Example Notebook <examples/pal.html>`_

**SQLDatabase Chain**

- **Links Used**: SQLDatabase, LLMChain
- **Notes**: This chain takes user input (a question), uses a first LLM chain to construct a SQL query to run against the SQL database, and then uses another LLMChain to take the results of that query and use it to answer the original question.
- `Example Notebook <examples/sqlite.html>`_

**LLMBash Chain**

- **Links Used**: BashProcess, LLMChain
- **Notes**: This chain takes user input (a question), uses an LLM chain to convert it to a bash command to run in the terminal, and then returns that as the result.
- `Example Notebook <examples/llm_bash.html>`_

**LLMChecker Chain**

- **Links Used**: LLMChain
- **Notes**: This chain takes user input (a question), uses an LLM chain to answer that question, and then uses other LLMChains to self-check that answer.
- `Example Notebook <examples/llm_checker.html>`_

**LLMRequests Chain**

- **Links Used**: Requests, LLMChain
- **Notes**: This chain takes a URL and other inputs, uses Requests to get the data at that URL, and then passes that along with the other inputs into an LLMChain to generate a response. The example included shows how to ask a question to Google - it firsts constructs a Google url, then fetches the data there, then passes that data + the original question into an LLMChain to get an answer.
- `Example Notebook <examples/llm_requests.html>`_

**Moderation Chain**

- **Links Used**: LLMChain, ModerationChain
- **Notes**: This chain shows how to use OpenAI's content moderation endpoint to screen output, and shows how to connect this to an LLMChain.
- `Example Notebook <examples/moderation.html>`_

**Transformation Chain**

- **Links Used**: TransformationChain
- **Notes**: This chain shows how to use the Transformation Chain, which takes an arbitrary python function and applies it to inputs/outputs of other chains.
- `Example Notebook <examples/transformation.html>`_


CombineDocuments Chains
-----------------------

`Question Answering <combine_docs_examples/question_answering.html>`_: A walkthrough of how to use LangChain for question answering over specific documents.

`Question Answering with Sources <combine_docs_examples/qa_with_sources.html>`_: A walkthrough of how to use LangChain for question answering (with sources) over specific documents.

`Summarization <combine_docs_examples/summarize.html>`_: A walkthrough of how to use LangChain for summarization over specific documents.

`Vector DB Question Answering <combine_docs_examples/vector_db_qa.html>`_: A walkthrough of how to use LangChain for question answering over a vector database.

`Vector DB Question Answering with Sources <combine_docs_examples/vector_db_qa_with_sources.html>`_: A walkthrough of how to use LangChain for question answering (with sources) over a vector database.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Generic Chains
   :hidden:

   examples/*

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: CombineDocument Chains
   :hidden:

   combine_docs_examples/*
