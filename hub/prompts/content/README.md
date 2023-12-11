# Работа с контентом

- `spell_correction.yaml` - исправление грамматики, орфографии и пунктуации. Аргумент - `text`
- `spell_correstion_system.yaml` - систем-промпт для инициализации работы в режиме переводчика.

## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain.chat_models import GigaChat

llm = GigaChat(credentials=="...")
prompt = load_prompt('lc://prompts/content/spell_correction.yaml')
chain = prompt | llm
text = chain.invoke({"text": "искуственый - интилектможет исправить все ошибки в даном тексте вне зависимости от длинны"})
```

Результат:
```
Искусственный интеллект может, исправить все ошибки в данном тексте вне зависимости от длины.
```