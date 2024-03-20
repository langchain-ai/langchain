# Работа с контентом

- `spell_correction.yaml` - исправление грамматики, орфографии и пунктуации. Аргумент - `text`
- `translation.yaml` - system промпт для перевода на русский язык с других языков.

## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain_community.chat_models import GigaChat

llm = GigaChat(credentials="...")
prompt = load_prompt('lc://prompts/content/spell_correction.yaml')
chain = prompt | llm
text = chain.invoke({"text": "искуственый - интилектможет исправить все ошибки в даном тексте вне зависимости от длинны"})
```

Результат:
```
Искусственный интеллект может, исправить все ошибки в данном тексте вне зависимости от длины.
```
> [!NOTE]
> Примеры использования остальных промптов вы можете посмотреть [здесь](content_examples.ipynb)