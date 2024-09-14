Databricks
==========

> [Databricks](https://www.databricks.com/) Intelligence Platform is the world's first data intelligence platform powered by generative AI. Infuse AI into every facet of your business.

Databricks embraces the LangChain ecosystem in various ways:

1. 🚀 **Model Serving** - Access state-of-the-art LLMs, such as DBRX, Llama3, Mixtral, or your fine-tuned models on [Databricks Model Serving](https://www.databricks.com/product/model-serving), via a highly available and low-latency inference endpoint. LangChain provides LLM (`Databricks`), Chat Model (`ChatDatabricks`), and Embeddings (`DatabricksEmbeddings`) implementations, streamlining the integration of your models hosted on Databricks Model Serving with your LangChain applications.
2. 📃 **Vector Search** - [Databricks Vector Search](https://www.databricks.com/product/machine-learning/vector-search) is a serverless vector database seamlessly integrated within the Databricks Platform. Using `DatabricksVectorSearch`, you can incorporate the highly scalable and reliable similarity search engine into your LangChain applications.
3. 📊 **MLflow** - [MLflow](https://mlflow.org/) is an open-source platform to manage full the ML lifecycle, including experiment management, evaluation, tracing, deployment, and more. [MLflow's LangChain Integration](/docs/integrations/providers/mlflow_tracking) streamlines the process of developing and operating modern compound ML systems.
4. 🌐 **SQL Database** - [Databricks SQL](https://www.databricks.com/product/databricks-sql) is integrated with `SQLDatabase` in LangChain, allowing you to access the auto-optimizing, exceptionally performant data warehouse.
5. 💡 **Open Models** - Databricks open sources models, such as [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), which are available through the [Hugging Face Hub](https://huggingface.co/databricks/dbrx-instruct). These models can be directly utilized with LangChain, leveraging its integration with the `transformers` library.

Installation
------------

First-party Databricks integrations are available in the langchain-databricks partner package.

```
pip install langchain-databricks
```

Chat Model
----------

`ChatDatabricks` is a Chat Model class to access chat endpoints hosted on Databricks, including state-of-the-art models such as Llama3, Mixtral, and DBRX, as well as your own fine-tuned models.

```
from langchain_databricks import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
```

See the [usage example](/docs/integrations/chat/databricks) for more guidance on how to use it within your LangChain application.

LLM
---

`Databricks` is an LLM class to access completion endpoints hosted on Databricks.

:::caution
Text completion models have been deprecated and the latest and most popular models are [chat completion models](/docs/concepts/#chat-models). Use `ChatDatabricks` chat model instead to use those models and advanced features such as tool calling.
:::

```
from langchain_community.llm.databricks import Databricks

llm = Databricks(endpoint="your-completion-endpoint")
```

See the [usage example](/docs/integrations/llms/databricks) for more guidance on how to use it within your LangChain application.


Embeddings
----------

`DatabricksEmbeddings` is an Embeddings class to access text-embedding endpoints hosted on Databricks, including state-of-the-art models such as BGE, as well as your own fine-tuned models.

```
from langchain_databricks import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
```

See the [usage example](/docs/integrations/text_embedding/databricks) for more guidance on how to use it within your LangChain application.


Vector Search
-------------

Databricks Vector Search is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from [Delta](https://docs.databricks.com/en/introduction/delta-comparison.html) tables managed by [Unity Catalog](https://www.databricks.com/product/unity-catalog) and query them with a simple API to return the most similar vectors.

```
from langchain_databricks.vectorstores import DatabricksVectorSearch

dvs = DatabricksVectorSearch(
    endpoint="<YOUT_ENDPOINT_NAME>",
    index_name="<YOUR_INDEX_NAME>",
    index,
    text_column="text",
    embedding=embeddings,
    columns=["source"]
)
docs = dvs.similarity_search("What is vector search?)
```

See the [usage example](/docs/integrations/vectorstores/databricks_vector_search) for how to set up vector indices and integrate them with LangChain.


MLflow Integration
------------------

In the context of LangChain integration, MLflow provides the following capabilities:

- **Experiment Tracking**: Tracks and stores models, artifacts, and traces from your LangChain experiments.
- **Dependency Management**: Automatically records dependency libraries, ensuring consistency among development, staging, and production environments.
- **Model Evaluation** Offers native capabilities for evaluating LangChain applications.
- **Tracing**: Visually traces data flows through your LangChain application.

See [MLflow LangChain Integration](/docs/integrations/providers/mlflow_tracking) to learn about the full capabilities of using MLflow with LangChain through extensive code examples and guides.

SQLDatabase
-----------
You can connect to Databricks SQL using the SQLDatabase wrapper of LangChain.
```
from langchain.sql_database import SQLDatabase

db = SQLDatabase.from_databricks(catalog="samples", schema="nyctaxi")
```

See [Databricks SQL Agent](https://docs.databricks.com/en/large-language-models/langchain.html#databricks-sql-agent) for how to connect Databricks SQL with your LangChain Agent as a powerful querying tool.

Open Models
-----------

To directly integrate Databricks's open models hosted on HuggingFace, you can use the [HuggingFace Integration](/docs/integrations/platforms/huggingface) of LangChain.

```
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="databricks/dbrx-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
llm.invoke("What is DBRX model?")
```
