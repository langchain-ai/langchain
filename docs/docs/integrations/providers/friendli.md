# Friendli AI

>[FriendliAI](https://friendli.ai/) enhances AI application performance and optimizes 
> cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.

## Installation and setup

Install the `friendli-client` python package.

```bash
pip install friendli-client
```
Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, 
and set it as the `FRIENDLI_TOKEN` environment variable.


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
