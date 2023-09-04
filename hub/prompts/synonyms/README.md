# Описания промптов для генерации синонимов

Промпты для генерации синонимов, синонимичных фраз и интентов для классических чат-ботов на правилах.

## Inputs
, dataset_size_max, subject, examples
1. `dataset_size_min`: минимальный размер датасета для генерации
2. `dataset_size_min`: максимальный размер датасета для генерации
3. `subject`: тема, для которой генерируются синонимы
4. `examples` (для промптов с примерами): примеры слов, которые должны быть в сгенерированных синонимах


## Usage

Below is a code snippet for how to use the prompt.

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