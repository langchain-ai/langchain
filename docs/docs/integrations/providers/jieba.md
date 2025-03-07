# langchain-jieba
An integration package connecting Jieba and LangChain

# Jieba link extractor

## Description
langchain_community.graph_vectorstores support storing texts and their relationships in a vector database in the form of a knowledge graph.But the
recommended extractor such as `KeybertLinkExtractor` works better in an English environment,while it performs poorly in a Chinese environment.
`JiebaLinkExtractor` extract keywords from texts depend on `jieba.analyse` lib, which more suitable in a Chinese environment.

## Installation
%pip install langchain_jieba

## Usage

### extract from single text

```python
from langchain_jieba import JiebaLinkExtractor

extractor = JiebaLinkExtractor()
results = extractor.extract_one("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(f"results={results}")
```
results={Link(kind='kw', direction='bidir', tag='日本京都大学'), Link(kind='kw', direction='bidir', tag='计算所'), Link(kind='kw', direction='bidir', tag='小明')}


### extract from multiple texts

```python
from langchain_jieba import JiebaLinkExtractor

CONTENT1 = '小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
CONTENT2 = '我来到北京清华大学'
extractor = JiebaLinkExtractor()
results = list(extractor.extract_many([CONTENT1, CONTENT2]))
print(f"results[0]={results[0]}")
print(f"results[1]={results[1]}")
```
results[0]={Link(kind='kw', direction='bidir', tag='日本京都大学'), Link(kind='kw', direction='bidir', tag='计算所'), Link(kind='kw', direction='bidir', tag='小明')}
results[1]={Link(kind='kw', direction='bidir', tag='清华大学'), Link(kind='kw', direction='bidir', tag='来到'), Link(kind='kw', direction='bidir', tag='北京')}