# 🦜️🔗 langchain_azure_ai


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

You can install the `langchain-azure-ai` package from PyPI.

```bash
pip install -U langchain-azure-ai
```

## Usage

Here's are some examples of how to use the `langchain-azure-ai` package.

1. Access the AzureOpenAI API: 

```python
from langchain_azure_ai.llms import AzureOpenAI

# Initialize the AzureOpenAI with gpt 3-5
llm = AzureOpenAI(
    deployment_name="gpt-35-turbo-instruct-0914",
)

# Invoke the llm
response = llm.invoke("Hello! How are you today?")
print(response)
```

2. Use the AzureSearch vector store: 

```python
from langchain_azure_ai.vectorstores.azuresearch import AzureSearch 
from langchain_azure_ai.embeddings import AzureOpenAIEmbeddings

vector_store_address: str = "YOUR_AZURE_SEARCH_ENDPOINT"
vector_store_password: str = "YOUR_AZURE_SEARCH_ADMIN_KEY"

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

index_name= "langchain-vector-demo" 
 
vector_store = AzureSearch( 
 azure_search_endpoint=vector_store_address, 
 azure_search_key=vector_store_password, 
 index_name=index_name, 
 embedding_function=embeddings.embed_query, 
)  
```


## Contributing

We welcome contributions to this project! Please follow the [contribution guide]() for instructions to setup the project for development and guidance on how to contribute effectively.

## License

This project is licensed under the [MIT License](LICENSE).
