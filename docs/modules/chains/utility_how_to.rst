Utility Chains
--------------

A chain is made up of links, which can be either primitives or other chains.
Primitives can be either `prompts <../prompts.html>`_, `llms <../llms.html>`_, `utils <../utils.html>`_, or other chains.
The examples here are all end-to-end chains for specific applications, focused on interacting an LLMChain with a specific utility.

**LLMMath**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a math question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Example Notebook <./examples/llm_math.html>`_

**PAL**

- **Links Used**: Python REPL, LLMChain
- **Notes**: This chain takes user input (a reasoning question), uses an LLMChain to convert it to python code snippet to run in the Python REPL, and then returns that as the result.
- `Paper <https://arxiv.org/abs/2211.10435>`_
- `Example Notebook <./examples/pal.html>`_

**SQLDatabase Chain**

- **Links Used**: SQLDatabase, LLMChain
- **Notes**: This chain takes user input (a question), uses a first LLM chain to construct a SQL query to run against the SQL database, and then uses another LLMChain to take the results of that query and use it to answer the original question.
- `Example Notebook <./examples/sqlite.html>`_

**API Chain**

- **Links Used**: LLMChain, Requests
- **Notes**: This chain first uses a LLM to construct the url to hit, then makes that request with the Requests wrapper, and finally runs that result through the language model again in order to product a natural language response.
- `Example Notebook <./examples/api.html>`_

**LLMBash Chain**

- **Links Used**: BashProcess, LLMChain
- **Notes**: This chain takes user input (a question), uses an LLM chain to convert it to a bash command to run in the terminal, and then returns that as the result.
- `Example Notebook <./examples/llm_bash.html>`_

**LLMChecker Chain**

- **Links Used**: LLMChain
- **Notes**: This chain takes user input (a question), uses an LLM chain to answer that question, and then uses other LLMChains to self-check that answer.
- `Example Notebook <./examples/llm_checker.html>`_

**LLMRequests Chain**

- **Links Used**: Requests, LLMChain
- **Notes**: This chain takes a URL and other inputs, uses Requests to get the data at that URL, and then passes that along with the other inputs into an LLMChain to generate a response. The example included shows how to ask a question to Google - it firsts constructs a Google url, then fetches the data there, then passes that data + the original question into an LLMChain to get an answer.
- `Example Notebook <./examples/llm_requests.html>`_

**Moderation Chain**

- **Links Used**: LLMChain, ModerationChain
- **Notes**: This chain shows how to use OpenAI's content moderation endpoint to screen output, and shows how to connect this to an LLMChain.
- `Example Notebook <./examples/moderation.html>`_


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Generic Chains
   :name: generic
   :hidden:

   ./examples/*