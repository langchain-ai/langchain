# Привет, мир!

Шаблон простого промпта, в ответ на который GigaChat возвращает фразу «Привет, мир!».

## Входные переменные

Шаблон не использует входных данных.

## Использование

Пример вызова:

```python
from langchain.prompts import load_prompt
from langchain_community.chat_models import GigaChat
from langchain.chains import LLMChain

giga = GigaChat(credentials="...")
prompt = load_prompt('lc://prompts/hello_world/prompt.yaml')
chain = LLMChain(llm=giga, prompt=prompt)
```
