# Baseten

[Baseten](https://baseten.co) provides all the infrastructure you need to deploy and serve ML models performantly, scalably, and cost-efficiently.

As a model inference platform, Baseten is a `Provider` in the LangChain ecosystem. The Baseten integration currently implements a single `Component`, LLMs, but more are planned!

Baseten lets you run both open source models like Llama 2 or Mistral and run proprietary or fine-tuned models on dedicated GPUs. If you're used to a provider like OpenAI, using Baseten has a few differences:

* Rather than paying per token, you pay per minute of GPU used.
* Every model on Baseten uses [Truss](https://truss.baseten.co/welcome), our open-source model packaging framework, for maximum customizability.
* While we have some [OpenAI ChatCompletions-compatible models](https://docs.baseten.co/api-reference/openai), you can define your own I/O spec with Truss.

You can learn more about Baseten in [our docs](https://docs.baseten.co/) or read on for LangChain-specific info.

## Setup: LangChain + Baseten

You'll need two things to use Baseten models with LangChain:

- A [Baseten account](https://baseten.co)
- An [API key](https://docs.baseten.co/observability/api-keys)

Export your API key to your as an environment variable called `BASETEN_API_KEY`.

```sh
export BASETEN_API_KEY="paste_your_api_key_here"
```

## Component guide: LLMs

Baseten integrates with LangChain through the [LLM component](https://python.langchain.com/docs/integrations/llms/baseten), which provides a standardized and interoperable interface for models that are deployed on your Baseten workspace.

You can deploy foundation models like Mistral and Llama 2 with one click from the [Baseten model library](https://app.baseten.co/explore/) or if you have your own model, [deploy it with Truss](https://truss.baseten.co/welcome).

In this example, we'll work with Mistral 7B. [Deploy Mistral 7B here](https://app.baseten.co/explore/mistral_7b_instruct) and follow along with the deployed model's ID, found in the model dashboard.

To use this module, you must:

* Export your Baseten API key as the environment variable BASETEN_API_KEY
* Get the model ID for your model from your Baseten dashboard
* Identify the model deployment ("production" for all model library models)

[Learn more](https://docs.baseten.co/deploy/lifecycle) about model IDs and deployments.

Production deployment (standard for model library models)

```python
from langchain_community.llms import Baseten

mistral = Baseten(model="MODEL_ID", deployment="production")
mistral("What is the Mistral wind?")
```

Development deployment

```python
from langchain_community.llms import Baseten

mistral = Baseten(model="MODEL_ID", deployment="development")
mistral("What is the Mistral wind?")
```

Other published deployment

```python
from langchain_community.llms import Baseten

mistral = Baseten(model="MODEL_ID", deployment="DEPLOYMENT_ID")
mistral("What is the Mistral wind?")
```

Streaming LLM output, chat completions, embeddings models, and more are all supported on the Baseten platform and coming soon to our LangChain integration. Contact us at [support@baseten.co](mailto:support@baseten.co) with any questions about using Baseten with LangChain.
