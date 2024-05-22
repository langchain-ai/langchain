# Testing the replace_imports migration

This runs the v0.2 migration with a desired set of rules.

```grit
language python

langchain_all_migrations()
```

## Single import

Before:

```python
from langchain.chat_models import ChatOpenAI
```

After:

```python
from langchain_community.chat_models import ChatOpenAI
```

## Community to partner

```python
from langchain_community.chat_models import ChatOpenAI
```

```python
from langchain_openai import ChatOpenAI
```

## Noop

This file should not match at all.

```python
from foo import ChatOpenAI
```

## Mixed imports

```python
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic, foo
```

```python
from langchain_community.chat_models import foo

from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic

```
