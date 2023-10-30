# RAG LangServe chain template

A basic chain template showing the RAG pattern using
a vector store on Astra DB / Apache Cassandra®.

## Setup:

You need:

- an [Astra](https://astra.datastax.com) Vector Database (free tier is fine!). **You need a [Database Administrator token](https://awesome-astra.github.io/docs/pages/astra/create-token/#c-procedure)**, in particular the string starting with `AstraCS:...`;
- likewise, get your [Database ID](https://awesome-astra.github.io/docs/pages/astra/faq/#where-should-i-find-a-database-identifier) ready, you will have to enter it below;
- an **OpenAI API Key**. (More info [here](https://cassio.org/start_here/#llm-access), note that out-of-the-box this demo supports OpenAI unless you tinker with the code.)

_Note:_ you can alternatively use a regular Cassandra cluster: to do so, make sure you provide the `USE_CASSANDRA_CLUSTER` entry as shown in `.env.template` and the subsequent environment variables to specify how to connect to it.

You need to provide the connection parameters and secrets through environment variables. Please refer to `.env.template` for what variables are required.

### Populate the vector store

Make sure you have the environment variables all set (see previous section),
then, from this directory, launch the following just once:

```
poetry run bash -c "cd [...]/cassandra_entomology_rag; python setup.py"
```

The output will be something like `Done (29 lines inserted).`.

> **Note**: In a full application, the vector store might be populated in other ways:
> this step is to pre-populate the vector store with some rows for the
> demo RAG chains to sensibly work.

### Sample inputs

The chain's prompt is engineered to stay on topic and only use the provided context.

To put this to test, experiment with these example questions:

```
"Are there more coleoptera or bugs?"
"Do Odonata have wings?"
"Do birds have wings?"                <-- no entomology here!
```

## Reference

Stand-alone repo with LangServe chain: [here](https://github.com/hemidactylus/langserve_cassandra_entomology_rag).
