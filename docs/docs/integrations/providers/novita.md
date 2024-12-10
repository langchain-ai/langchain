# Novita AI

>[Novita AI](https://novita.ai) is a generative AI inference platform to run and 
> customize models with industry-leading speed and production-readiness.



## Installation and setup

- Get a Novita AI API key by signing up at [novita.ai](https://novita.ai).
- Authenticate by setting the NOVITA_API_KEY environment variable.

### Authentication

There are two ways to authenticate using your Novita API key:

1.  Setting the `NOVITA_API_KEY` environment variable.

    ```python
    os.environ["NOVITA_API_KEY"] = "<KEY>"
    ```

2.  Setting `api_key` field in the Novita LLM module.

    ```python
    llm = Novita(api_key="<KEY>")
    ```
## Chat models

See a [usage example](/docs/integrations/chat/novita).

```python
from langchain_community.chat_models import ChatNovita
```
