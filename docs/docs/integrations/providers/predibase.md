# Predibase

Learn how to use LangChain with models on Predibase. 

## Setup
- Create a [Predibase](https://predibase.com/) account and [API key](https://docs.predibase.com/sdk-guide/intro).
- Install the Predibase Python client with `pip install predibase`
- Use your API key to authenticate

### LLM

Predibase integrates with LangChain by implementing LLM module. You can see a short example below or a full notebook under LLM > Integrations > Predibase. 

```python
import os
os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

from langchain_community.llms import Predibase

model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
)

response = model.invoke("Can you recommend me a nice dry wine?")
print(response)
```

Predibase also supports Predibase-hosted and HuggingFace-hosted adapters that are fine-tuned on the base model given by the `model` argument:

```python
import os
os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

from langchain_community.llms import Predibase

# The fine-tuned adapter is hosted at Predibase (adapter_version must be specified).
model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
)

response = model.invoke("Can you recommend me a nice dry wine?")
print(response)
```

Predibase also supports adapters that are fine-tuned on the base model given by the `model` argument:

```python
import os
os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

from langchain_community.llms import Predibase

# The fine-tuned adapter is hosted at HuggingFace (adapter_version does not apply and will be ignored).
model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
)

response = model.invoke("Can you recommend me a nice dry wine?")
print(response)
```
