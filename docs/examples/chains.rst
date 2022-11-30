Chains
======

The examples here are all end-to-end chains for specific applications.
A chain consists of multiple primitives called in a deterministic manner.

The following primitives exist as options:

#. `LLM: <../modules/llms.rst>`_ A language model takes text as input and outputs text.
#. `PromptTemplate: <../modules/prompt.rst>`_ A prompt template takes arbitrary string inputs and returns a final formatted string.
#. `Python REPL: <../modules/python.rst>`_ A Python REPL takes a string representing a Python command to run, runs that command, and then returns anything that was printed during that run.
#. `SQL Database: <../modules/sql_database.rst>`_ A SQL database takes a string representing a SQL command as input and executes that command against the database. If any rows are returned, then those are cast to a string and returned.
#. `Search: <../modules/search.rst>`_ A search object takes a string as input and executes that against a search object, returning any results.
#. `Docstore: <../modules/docstore.rst>`_ A docstore object can be used to lookup a document in a database by exact match.
#. `Vectorstore: <../modules/vectorstore.rst>`_ A vectorstore object uses embeddings stored in a vector database to take in an input string and return documents similar to that string.


With these primitives in mind, the following chains exist:

**LLMChain**

- **Primitives Used**: PromptTemplate, LLM
- **Notes**: This chain is the simplest chain, and is widely used by almost every other chain. This chain takes arbitrary user input, creates a prompt with it from the PromptTemplate, passes that to the LLM, and then returns the output of the LLM as the final output.
- **Description**: Use case really depends on the PromptTemplate you use
- `Example Notebook <chains/llm_chain>`_



.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Chains
   :hidden:

   chains/*
