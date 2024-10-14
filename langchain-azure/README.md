# 🦜️🔗 LangChain 🤝 Amazon Web Services (AWS)

This repository provides LangChain components for various Azure services. 

## Features

- **agent_toolkits**
- **chat_models**
- **document_loaders**
- **embeddings**
- **llms**
- **retrievers**
- **tools**
- **vectorstores**

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Installation

You can install the `langchain-azure` package from PyPI.

```bash
pip install langchain-azure
```

## Usage

Here's a simple example of how to use the `langchain-azure` package.

```python
from langchain_azure import AzureOpenAI

# Initialize the Bedrock LLM
llm = AzureOpenAI(
    deployment_name="gpt-35-turbo-instruct-0914",
)

# Invoke the llm
response = llm.invoke("Hello! How are you today?")
print(response)
```


## Contributing

We welcome contributions to this project! Please follow the [contribution guide]() for instructions to setup the project for development and guidance on how to contribute effectively.

## License

This project is licensed under the [MIT License](LICENSE).
