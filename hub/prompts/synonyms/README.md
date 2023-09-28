# Создание синонимов

Шаблоны промптов для создания синонимов, синонимичных фраз, а так же интентов для сценариев чат-ботов и виртуальных ассистентов.

## Входные переменные

В зависимости от шаблона на вход промптам можно передавать следующие переменные:

- `dataset_size_min` — минимальный размер датасета для генерации;
- `dataset_size_max` — максимальный размер датасета для генерации;
- `subject` — тема, для которой генерируются синонимы;
- `examples` — примеры слов, которые должны быть в сгенерированных синонимах. Используется в шаблонах с примерами.


## Использование

Пример генерации синонимов:

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain

llm = ...
synonyms_with_examples = load_prompt('lc://prompts/synonyms/synonyms_generation_with_examples.yaml')
text = prompt.format(dataset_size_min=5,
                        dataset_size_max=10,
                        subject="кошка",
                        examples='["кот", "котёнок"]')
```