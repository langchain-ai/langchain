# Суммаризация по алгоритму MapReduce

Шаблоны, предназначенные для суммаризации текстов с помощью GigaChat по алгоритму MapReduce.

## Входные переменные

На вход в промпты нужно передать переменную `text`, которая содержит текст, подлежащий суммаризации.

## Использование

Пример вызова суммаризатора:

```python
from langchain.prompts import load_prompt
from langchain.chains.summarize import load_summarize_chain

giga = GigaChat(credentials="...")
map_prompt = load_prompt('lc://prompts/summarize/map_reduce/map.yaml')
combine_prompt = load_prompt('lc://prompts/summarize/map_reduce/combine.yaml')

chain = load_summarize_chain(giga, chain_type="map_reduce", map_prompt=map_prompt,
            combine_prompt=combine_prompt)
```