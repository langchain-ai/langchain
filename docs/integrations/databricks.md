Databricks
==========

The [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform.

Databricks embraces the LangChain ecosystem in various ways:
1. Databricks as an LLM provider
2. Databricks Dolly can be accessed through HuggingFace Hub
3. Databricks runtimes and Databricks SQL can be accessed using the SQLDatabase wrapper
4. Databricks-host MLflow makes it easier to track and serve LangChain applications

Databricks as an LLM provider
-----------------------------

The notebook [Wrap Databricks endpoints as LLMs](../modules/models/llms/integrations/databricks.html) illustrates the method to wrap Databricks endpoints as LLMs in LangChain. It supports two types of endpoints: the serving endpoint, which is recommended for both production and development, and the cluster driver proxy app, which is recommended for interactive development. 

Databricks Dolly
----------------

Databricks’ Dolly is an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. The model is available on Hugging Face Hub as databricks/dolly-v2-12b. See the notebook [HuggingFace Hub](../modules/models/llms/integrations/huggingface_hub.html) for instructions to access it through the HuggingFace Hub integration with LangChain. 

Databricks runtimes and Databricks SQL
--------------------------------------
You can connect to [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) the SQLDatabase wrapper of LangChain. See the notebook [Connect to Databricks](./databricks/databricks.html) for details.


Managed MLflow on Databricks
----------------------------
Managed MLflow is built on top of MLflow, an open source platform developed by Databricks to help manage the complete machine learning lifecycle with enterprise reliability, security and scale. Managed MLflow on Databricks is a fully managed version of MLflow providing practitioners with reproducibility and experiment management across Databricks Notebooks, Jobs, and data stores, with the reliability, security, and scalability of the Databricks Lakehouse Platform.

On Databricks, you don't need to set the tracking uri.
