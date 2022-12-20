Chains
======

The examples here are all end-to-end chains for specific applications.
A chain is made up of links, which can be either primitives or other chains.

The following primitives exist as options to use for links:

#. `LLM: <../modules/llms.rst>`_ A language model takes text as input and outputs text.
#. `PromptTemplate: <../modules/prompt.rst>`_ A prompt template takes arbitrary string inputs and returns a final formatted string.
#. `Python REPL: <../modules/python.rst>`_ A Python REPL takes a string representing a Python command to run, runs that command, and then returns anything that was printed during that run.
#. `SQL Database: <../modules/sql_database.rst>`_ A SQL database takes a string representing a SQL command as input and executes that command against the database. If any rows are returned, then those are cast to a string and returned.
#. `Search: <../modules/serpapi.rst>`_ A search object takes a string as input and executes that against a search object, returning any results.

With these primitives in mind, the following chains exist:

**LLMChain**

- **Links Used**: PromptTemplate, LLM
- **Notes**: This chain is the simplest chain, and is widely used by almost every other chain. This chain takes arbitrary user input, creates a prompt with it from the PromptTemplate, passes that to the LLM, and then returns the output of the LLM as the final output.
- `Example Notebook <chains/llm_chain.ipynb>`_

**LLMMath**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a math question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Example Notebook <chains/llm_math.ipynb>`_

**PAL**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a reasoning question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Paper <https://arxiv.org/abs/2211.10435>`_
- `Example Notebook <chains/pal.ipynb>`_

**SQLDatabase Chain**

- **Links Used**: SQLDatabase, LLMChain
- **Notes**: This chain takes user input (a question), uses a first LLM chain to construct a SQL query to run against the SQL database, and then uses another LLMChain to take the results of that query and use it to answer the original question.
- `Example Notebook <chains/sqlite.ipynb>`_


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Chains
   :hidden:

   chains/*
