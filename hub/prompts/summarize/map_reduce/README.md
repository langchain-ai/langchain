# Описания промптов для map-reduce суммаризации

Данные промпты хорошо подходят для построения суммаризацтора, использующего алгоритм map-reduce.

## Inputs

1. `text`: Текст, который нужно суммаризировать


## Usage

Below is a code snippet for how to use the prompt.

```python
from langchain.prompts import load_prompt
from langchain.chains.summarize import load_summarize_chain

llm = ...
map_prompt = load_prompt('lc://prompts/summarize/map_reduce/map.yaml')
combine_prompt = load_prompt('lc://prompts/summarize/map_reduce/combine.yaml')

chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt,
            combine_prompt=combine_prompt)
```