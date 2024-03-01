# LangChain-Fireworks

This is the partner package for tying Fireworks.ai and LangChain. Fireworks really strive to provide good support for LangChain use cases, so if you run into any issues please let us know. You can reach out to us [in our Discord channel](https://discord.com/channels/1137072072808472616/)


## Installation

To use the `langchain-fireworks` package, follow these installation steps:

```bash
pip install langchain-fireworks
```



## Basic usage

### Setting up

1. Sign in to [Fireworks AI](http://fireworks.ai/) to obtain an API Key to access the models, and make sure it is set as the `FIREWORKS_API_KEY` environment variable.

    Once you've signed in and obtained an API key, follow these steps to set the `FIREWORKS_API_KEY` environment variable:
    - **Linux/macOS:** Open your terminal and execute the following command:
    ```bash
    export FIREWORKS_API_KEY='your_api_key'
    ```
    **Note:** To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

    - **Windows:** For Command Prompt, use:
    ```cmd
    set FIREWORKS_API_KEY=your_api_key
    ```

2. Set up your model using a model id. If the model is not set, the default model is `fireworks-llama-v2-7b-chat`. See the full, most up-to-date model list on [fireworks.ai](https://fireworks.ai/models).

```python
import getpass
import os

# Initialize a Fireworks model
llm = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
)
```


### Calling the Model Directly

You can call the model directly with string prompts to get completions.

```python
# Single prompt
output = llm.invoke("Who's the best quarterback in the NFL?")
print(output)
```

```python
# Calling multiple prompts
output = llm.generate(
    [
        "Who's the best cricket player in 2016?",
        "Who's the best basketball player in the league?",
    ]
)
print(output.generations)
```





## Advanced usage
### Tool use: LangChain Agent + Fireworks function calling model
Please checkout how to teach Fireworks function calling model to use a [calculator here](https://github.com/fw-ai/cookbook/blob/main/examples/function_calling/fireworks_langchain_tool_usage.ipynb). 

Fireworks focus on delivering the best experience for fast model inference as well as tool use. You can check out [our blog](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling) for more details on how it fares compares to GPT-4, the punchline is that it is on par with GPT-4 in terms just function calling use cases, but it is way faster and much cheaper.

### RAG: LangChain agent + Fireworks function calling model + MongoDB + Nomic AI embeddings
Please check out the [cookbook here](https://github.com/fw-ai/cookbook/blob/main/examples/rag/mongodb_agent.ipynb) for an end to end flow