# RAG with Timescale Vector using hybrid search

This template shows how to use timescale-vector with the self-query retriver to perform hybrid search on similarity and time.
This is useful any time your data has a strong time-based component. Some examples of such data are:
- News articles (politics, business, etc)
- Blog posts, documentation or other published material (public or private).
- Social media posts
- Changelogs of any kind
- Messages

Such items are often searched by both similarity and time. For example: Show me all news about Toyota trucks from 2022.

[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral)  provides superior performance when searching for embeddings within a particular
timeframe by leveraging automatic table partitioning to isolate data for particular time-ranges.

Langchain's self-query retriever allows deducing time-ranges (as well as other search criteria) from the text of user queries.

## What is Timescale Vector?
**[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) is PostgreSQL++ for AI applications.**

Timescale Vector enables you to efficiently store and query billions of vector embeddings in `PostgreSQL`.
- Enhances `pgvector` with faster and more accurate similarity search on 1B+ vectors via DiskANN inspired indexing algorithm.
- Enables fast time-based vector search via automatic time-based partitioning and indexing.
- Provides a familiar SQL interface for querying vector embeddings and relational data.

Timescale Vector is cloud PostgreSQL for AI that scales with you from POC to production:
- Simplifies operations by enabling you to store relational metadata, vector embeddings, and time-series data in a single database.
- Benefits from rock-solid PostgreSQL foundation with enterprise-grade feature liked streaming backups and replication, high-availability and row-level security.
- Enables a worry-free experience with enterprise-grade security and compliance.

### How to access Timescale Vector
Timescale Vector is available on [Timescale](https://www.timescale.com/products?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral), the cloud PostgreSQL platform. (There is no self-hosted version at this time.)

- LangChain users get a 90-day free trial for Timescale Vector.
- To get started, [signup](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) to Timescale, create a new database and follow this notebook!
- See the [installation instructions](https://github.com/timescale/python-vector) for more details on using Timescale Vector in python.

### Using Timescale Vector with this template

This template uses TimescaleVector as a vectorstore and requires that `TIMESCALES_SERVICE_URL` is set.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## Loading sample data

We have provided a sample dataset you can use for demoing this template. It consists of the git history of the timescale project.

To load this dataset, set the `LOAD_SAMPLE_DATA` environmental variable.

## Loading your own dataset.

To load your own dataset you will have to modify the code in the `DATASET SPECIFIC CODE` section of `chain.py`.
This code defines the name of the collection, how to load the data, and the human-language description of both the
contents of the collection and all of the metadata. The human-language descriptions are used by the self-query retriever
to help the LLM convert the question into filters on the metadata when searching the data in Timescale-vector.

## Using in your own applications

This is a standard LangServe template. Instructions on how to use it with your LangServe applications are [here](https://github.com/langchain-ai/langchain/blob/master/templates/README.md).


