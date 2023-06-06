# Amazon Bedrock

>[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that makes FMs from leading AI startups and Amazon available via an API, so you can choose from a wide range of FMs to find the model that is best suited for your use case.

## Installation and Setup

```bash
pip install boto3
```

## LLM

See a [usage example](../modules/models/llms/integrations/bedrock.ipynb).

```python
from langchain import Bedrock
```

## Text Embedding Models

See a [usage example](../modules/models/text_embedding/examples/amazon_bedrock.ipynb).
```python
from langchain.embeddings import BedrockEmbeddings
```
