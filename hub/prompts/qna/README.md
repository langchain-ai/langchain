# Поиск по документам (вопрос-ответ)

Шаблоны промптов для генерации ответов на вопросы по переданным документам:

- `qna_with_refs_system.yaml` — поиск ответа на вопрос по документам со сслыками на источники (system-часть).
- `qna_with_refs_user.yaml` — gоиск ответа на вопрос по документам со сслыками на источники (user-часть).
- `generate_question_prompt.yaml` — генерация вопросов к документу. Используется для улучшения качества индексации.

## Входные переменные

В зависимости от шаблона на вход промптам можно передавать следующие переменные:

- `question` — вопрос;
- `summaries` — данные, которые должны быть использованы в ответе.

## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain

llm = ...hub/
generate_question_prompt = load_prompt('lc://prompts/qna/generate_question_prompt.yaml')
text = generate_question_prompt.format(text="... text of your documents ...")
```
