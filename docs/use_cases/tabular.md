# Querying Tabular Data

> [Conceptual Guide](https://docs.langchain.com/docs/use-cases/qa-tabular)


Lots of data and information is stored in tabular data, whether it be csvs, excel sheets, or SQL tables.
This page covers all resources available in LangChain for working with data in this format.

## Document Loading
If you have text data stored in a tabular format, you may want to load the data into a Document and then index it as you would
other text/unstructured data. For this, you should use a document loader like the [CSVLoader](../modules/indexes/document_loaders/examples/csv.ipynb)
and then you should [create an index](../modules/indexes.rst) over that data, and [query it that way](../modules/chains/index_examples/vector_db_qa.ipynb).

## Querying
If you have more numeric tabular data, or have a large amount of data and don't want to index it, you should get started
by looking at various chains and agents we have for dealing with this data.

### Chains

If you are just getting started, and you have relatively small/simple tabular data, you should get started with chains.
Chains are a sequence of predetermined steps, so they are good to get started with as they give you more control and let you 
understand what is happening better.

- [SQL Database Chain](../modules/chains/examples/sqlite.ipynb)

### Agents

Agents are more complex, and involve multiple queries to the LLM to understand what to do.
The downside of agents are that you have less control. The upside is that they are more powerful,
which allows you to use them on larger databases and more complex schemas. 

- [SQL Agent](../modules/agents/toolkits/examples/sql_database.ipynb)
- [Pandas Agent](../modules/agents/toolkits/examples/pandas.ipynb)
- [CSV Agent](../modules/agents/toolkits/examples/csv.ipynb)
