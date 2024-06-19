# langchain-ibm

This package provides the integration between LangChain and IBM watsonx.ai through the `ibm-watsonx-ai` SDK.

## Installation

To use the `langchain-ibm` package, follow these installation steps:

```bash
pip install langchain-ibm
```

## Usage

### Setting up

To use IBM's models, you must have an IBM Cloud user API key. Here's how to obtain and set up your API key:

1. **Obtain an API Key:** For more details on how to create and manage an API key, refer to IBM's [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
2. **Set the API Key as an Environment Variable:** For security reasons, it's recommended to not hard-code your API key directly in your scripts. Instead, set it up as an environment variable. You can use the following code to prompt for the API key and set it as an environment variable:

```python
import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key
```

In alternative, you can set the environment variable in your terminal.

- **Linux/macOS:** Open your terminal and execute the following command:
     ```bash
     export WATSONX_APIKEY='your_ibm_api_key'
     ```
     To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

- **Windows:** For Command Prompt, use:
    ```cmd
    set WATSONX_APIKEY=your_ibm_api_key
    ```

### Loading the model

You might need to adjust model parameters for different models or tasks. For more details on the parameters, refer to IBM's [documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).

```python
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}
```

Initialize the WatsonxLLM class with the previously set parameters.

```python
from langchain_ibm import WatsonxLLM

watsonx_llm = WatsonxLLM(
    model_id="PASTE THE CHOSEN MODEL_ID HERE",
    url="PASTE YOUR URL HERE",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

**Note:**
- You must provide a `project_id` or `space_id`. For more information refer to IBM's [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
- Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
- You need to specify the model you want to use for inferencing through `model_id`. You can find the list of available models [here](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes).


Alternatively you can use Cloud Pak for Data credentials. For more details, refer to IBM's [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

```python
watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

### Create a Chain

Create `PromptTemplate` objects which will be responsible for creating a random question.

```python
from langchain.prompts import PromptTemplate

template = "Generate a random question about {topic}: Question: "
prompt = PromptTemplate.from_template(template)
```

Provide a topic and run the LLMChain.

```python
from langchain.chains import LLMChain

llm_chain = LLMChain(prompt=prompt, llm=watsonx_llm)
response = llm_chain.invoke("dog")
print(response)
```

### Calling the Model Directly
To obtain completions, you can call the model directly using a string prompt.

```python
# Calling a single prompt

response = watsonx_llm.invoke("Who is man's best friend?")
print(response)
```

```python
# Calling multiple prompts

response = watsonx_llm.generate(
    [
        "The fastest dog in the world?",
        "Describe your chosen dog breed",
    ]
)
print(response)
```

### Streaming the Model output

You can stream the model output.

```python
for chunk in watsonx_llm.stream(
    "Describe your favorite breed of dog and why it is your favorite."
):
    print(chunk, end="")
```
