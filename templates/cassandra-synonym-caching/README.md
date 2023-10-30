# LLM-cache LangServe chain template

A simple chain template showcasing usage of LLM Caching
backed by Astra DB / Apache Cassandra®.

## Setup:

You need:

- an [Astra](https://astra.datastax.com) Vector Database (free tier is fine!). **You need a [Database Administrator token](https://awesome-astra.github.io/docs/pages/astra/create-token/#c-procedure)**, in particular the string starting with `AstraCS:...`;
- likewise, get your [Database ID](https://awesome-astra.github.io/docs/pages/astra/faq/#where-should-i-find-a-database-identifier) ready, you will have to enter it below;
- an **OpenAI API Key**. (More info [here](https://cassio.org/start_here/#llm-access), note that out-of-the-box this demo supports OpenAI unless you tinker with the code.)

_Note:_ you can alternatively use a regular Cassandra cluster: to do so, make sure you provide the `USE_CASSANDRA_CLUSTER` entry as shown in `.env.template` and the subsequent environment variables to specify how to connect to it.

You need to provide the connection parameters and secrets through environment variables. Please refer to `.env.template` for what variables are required.

## Reference

Stand-alone LangServe template repo: [here](https://github.com/hemidactylus/langserve_cassandra_synonym_caching).
