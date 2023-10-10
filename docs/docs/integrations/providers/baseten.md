# Baseten

Learn how to use LangChain with models deployed on Baseten.

## Installation and setup

- Create a [Baseten](https://baseten.co) account and [API key](https://docs.baseten.co/settings/api-keys).
- Install the Baseten Python client with `pip install baseten`
- Use your API key to authenticate with `baseten login`

## Invoking a model

Baseten integrates with LangChain through the LLM module, which provides a standardized and interoperable interface for models that are deployed on your Baseten workspace.

You can deploy foundation models like WizardLM and Alpaca with one click from the [Baseten model library](https://app.baseten.co/explore/) or if you have your own model, [deploy it with this tutorial](https://docs.baseten.co/deploying-models/deploy).

In this example, we'll work with WizardLM. [Deploy WizardLM here](https://app.baseten.co/explore/wizardlm) and follow along with the deployed [model's version ID](https://docs.baseten.co/managing-models/manage).

```python
from langchain.llms import Baseten

wizardlm = Baseten(model="MODEL_VERSION_ID", verbose=True)

wizardlm("What is the difference between a Wizard and a Sorcerer?")
```
