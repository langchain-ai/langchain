# Промпт для MultiQueryRetriever

### insurance_agent.yaml
Генератор разных версий вопроса к вопросу


## Пример использования

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import JSONLineListOutputParser

llm = ...hub/ # желательно 70b
insurance_prompt = load_prompt(
    'lc://prompts/retrievers/multi_query/insurance_agent.yaml'
)
llm_chain = LLMChain(
    llm=llm,
    prompt=insurance_prompt,
    output_parser=JSONLineListOutputParser(),
)

print(llm_chain.run(question="Страхуются ли музыкальные инструменты?"))
```