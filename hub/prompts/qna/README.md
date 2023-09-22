# Промпты для поиска по документам (вопрос-ответ)

### qna_with_refs_system.yaml
Поиск ответа на вопрос по документам со сслыками на источники (system-часть)
### qna_with_refs_user.yaml
Поиск ответа на вопрос по документам со сслыками на источники (user-часть)
### generate_question_prompt.yaml
Генерация вопросов к документу. Используется для улучшения качества индексации.

## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain

llm = ...hub/
generate_question_prompt = load_prompt('lc://prompts/qna/generate_question_prompt.yaml')
text = generate_question_prompt.format(text="... text of your docuemtns ...")
```