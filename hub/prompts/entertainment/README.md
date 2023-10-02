# Промпты для поиска по документам (вопрос-ответ)

### meditation.yaml
Генератор медитаций на заданную тему


## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain

llm = ...hub/
meditation_prompt = load_prompt('lc://prompts/entertainment/meditation.yaml')
text = meditation_prompt.format(background="sea waves", topic="утро на море")
```