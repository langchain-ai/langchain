Databricks
==========

The [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform.

Databricks embraces the LangChain ecosystem in various ways:

1. Databricks connector for the SQLDatabase Chain: SQLDatabase.from_databricks() provides an easy way to query your data on Databricks through LangChain
2. Databricks MLflow integrates with LangChain: Tracking and serving LangChain applications with fewer steps
3. Databricks MLflow AI Gateway
4. Databricks as an LLM provider: Deploy your fine-tuned LLMs on Databricks via serving endpoints or cluster driver proxy apps, and query it as langchain.llms.Databricks
5. Databricks Dolly: Databricks open-sourced Dolly which allows for commercial use, and can be accessed through the Hugging Face Hub

Databricks connector for the SQLDatabase Chain
----------------------------------------------
You can connect to [Databricks runtimes](https://docs.databricks.com/runtime/index.html) and [Databricks SQL](https://www.databricks.com/product/databricks-sql) using the SQLDatabase wrapper of LangChain. 
See the notebook [Connect to Databricks](/docs/use_cases/qa_structured/integrations/databricks) for details.

Databricks MLflow integrates with LangChain
-------------------------------------------

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. See the notebook [MLflow Callback Handler](/docs/integrations/providers/mlflow_tracking) for details about MLflow's integration with LangChain.

Databricks provides a fully managed and hosted version of MLflow integrated with enterprise security features, high availability, and other Databricks workspace features such as experiment and run management and notebook revision capture. MLflow on Databricks offers an integrated experience for tracking and securing machine learning model training runs and running machine learning projects. See [MLflow guide](https://docs.databricks.com/mlflow/index.html) for more details.

Databricks MLflow makes it more convenient to develop LangChain applications on Databricks. For MLflow tracking, you don't need to set the tracking uri. For MLflow Model Serving, you can save LangChain Chains in the MLflow langchain flavor, and then register and serve the Chain with a few clicks on Databricks, with credentials securely managed by MLflow Model Serving.

Databricks MLflow AI Gateway
----------------------------

See [MLflow AI Gateway](/docs/integrations/providers/mlflow_ai_gateway).

Databricks as an LLM provider
-----------------------------

The notebook [Wrap Databricks endpoints as LLMs](/docs/integrations/llms/databricks) illustrates the method to wrap Databricks endpoints as LLMs in LangChain. It supports two types of endpoints: the serving endpoint, which is recommended for both production and development, and the cluster driver proxy app, which is recommended for interactive development. 

Databricks endpoints support Dolly, but are also great for hosting models like MPT-7B or any other models from the Hugging Face ecosystem. Databricks endpoints can also be used with proprietary models like OpenAI to provide a governance layer for enterprises.

Databricks Dolly
----------------

Databricks’ Dolly is an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. The model is available on Hugging Face Hub as databricks/dolly-v2-12b. See the notebook [Hugging Face Hub](/docs/integrations/llms/huggingface_hub) for instructions to access it through the Hugging Face Hub integration with LangChain. 
