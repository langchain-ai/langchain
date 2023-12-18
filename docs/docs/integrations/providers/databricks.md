Databricks
==========

The [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform.

Databricks embraces the LangChain ecosystem in various ways:

1. Databricks connector for the SQLDatabase Chain: SQLDatabase.from_databricks() provides an easy way to query your data on Databricks through LangChain
2. Databricks MLflow integrates with LangChain: Tracking and serving LangChain applications with fewer steps
3. Databricks as an LLM provider: Deploy your fine-tuned LLMs on Databricks via serving endpoints or cluster driver proxy apps, and query it as langchain.llms.Databricks
4. Databricks Dolly: Databricks open-sourced Dolly which allows for commercial use, and can be accessed through the Hugging Face Hub

Databricks connector for the SQLDatabase Chain
----------------------------------------------
You can connect to [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) using the SQLDatabase wrapper of LangChain. 
See the notebook [Connect to Databricks](/docs/use_cases/qa_structured/integrations/databricks) for details.

Databricks MLflow integrates with LangChain
-------------------------------------------

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. See the notebook [MLflow Callback Handler](/docs/integrations/providers/mlflow_tracking) for details about MLflow's integration with LangChain.

Databricks provides a fully managed and hosted version of MLflow integrated with enterprise security features, high availability, and other Databricks workspace features such as experiment and run management and notebook revision capture. MLflow on Databricks offers an integrated experience for tracking and securing machine learning model training runs and running machine learning projects. See [MLflow guide](https://docs.databricks.com/mlflow/index.html) for more details.

Databricks MLflow makes it more convenient to develop LangChain applications on Databricks. For MLflow tracking, you don't need to set the tracking uri. For MLflow Model Serving, you can save LangChain Chains in the MLflow langchain flavor, and then register and serve the Chain with a few clicks on Databricks, with credentials securely managed by MLflow Model Serving.

Databricks External Models
--------------------------

[Databricks External Models](https://docs.databricks.com/generative-ai/external-models/index.html) is a service that is designed to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization. It offers a high-level interface that simplifies the interaction with these services by providing a unified endpoint to handle specific LLM related requests. The following example creates an endpoint that serves OpenAI's GPT-4 model and generates a chat response from it:

```python
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client


client = get_deploy_client("databricks")
name = f"chat"
client.create_endpoint(
    name=name,
    config={
        "served_entities": [
            {
                "name": "test",
                "external_model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        "openai_api_key": "{{secrets/<scope>/<key>}}",
                    },
                },
            }
        ],
    },
)
chat = ChatDatabricks(endpoint=name, temperature=0.1)
print(chat([HumanMessage(content="hello")]))
# -> content='Hello! How can I assist you today?'
```

Databricks Foundation Model APIs
--------------------------------

[Databricks Foundation Model APIs](https://docs.databricks.com/machine-learning/foundation-models/index.html) allow you to access and query state-of-the-art open source models from dedicated serving endpoints. With Foundation Model APIs, developers can quickly and easily build applications that leverage a high-quality generative AI model without maintaining their own model deployment. The following example uses the `databricks-bge-large-en` endpoint to generate embeddings from  text:

```python
from langchain.embeddings import DatabricksEmbeddings


embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(embeddings.embed_query("hello")[:3])
# -> [0.051055908203125, 0.007221221923828125, 0.003879547119140625, ...]
```

Databricks as an LLM provider
-----------------------------

The notebook [Wrap Databricks endpoints as LLMs](/docs/integrations/llms/databricks#wrapping-a-serving-endpoint-custom-model) demonstrates how to serve a custom model that has been registered by MLflow as a Databricks endpoint.
It supports two types of endpoints: the serving endpoint, which is recommended for both production and development, and the cluster driver proxy app, which is recommended for interactive development. 


Databricks Vector Search
------------------------

Databricks Vector Search is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from Delta tables managed by Unity Catalog and query them with a simple API to return the most similar vectors. See the notebook [Databricks Vector Search](/docs/integrations/vectorstores/databricks_vector_search) for instructions to use it with LangChain.
