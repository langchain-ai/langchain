# elastic-query-generator

We can use LLMs to interact with Elasticsearch analytics databases in natural language.

This chain builds search queries via the Elasticsearch DSL API (filters and aggregations).

The Elasticsearch client must have permissions for index listing, mapping description and search queries.



## Setup

## Installing Elasticsearch

There are a number of ways to run Elasticsearch.

### Elastic Cloud

Create a free trial account on [Elastic Cloud](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=langserve).

With a deployment, update the connection string.

Password and connection (elasticsearch url) can be found on the deployment console. Th 

## Populating with data

If you want to populate the DB with some example info, you can run `python ingest.py`.

This will create a `customers` index.
In the chain, we specify indexes to generate queries against, and we specify `["customers"]`.
This is specific to setting up your Elastic index in this 
