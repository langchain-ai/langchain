# Friendli AI

>[Friendli AI](https://friendli.ai/) is a company that fine-tunes, deploys LLMs, 
> and serves a wide range of Generative AI use cases.


## Installation and setup

- Install the integration package:

  ```
  pip install friendli-client
  ```

- Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, 
and set it as the `FRIENDLI_TOKEN` environment.

## Chat models

See a [usage example](/docs/integrations/chat/friendli).

```python
from langchain_community.chat_models.friendli import ChatFriendli
```

## LLMs

See a [usage example](/docs/integrations/llms/friendli).

```python
from langchain_community.llms.friendli import Friendli
```
