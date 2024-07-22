# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

web_search_template = """Please write a passage to answer the question 
Question: {QUESTION}
Passage:"""
web_search = PromptTemplate(template=web_search_template, input_variables=["QUESTION"])
sci_fact_template = """Please write a scientific paper passage to support/refute the claim 
Claim: {Claim}
Passage:"""
sci_fact = PromptTemplate(template=sci_fact_template, input_variables=["Claim"])
arguana_template = """Please write a counter argument for the passage 
Passage: {PASSAGE}
Counter Argument:"""
arguana = PromptTemplate(template=arguana_template, input_variables=["PASSAGE"])
trec_covid_template = """Please write a scientific paper passage to answer the question
Question: {QUESTION}
Passage:"""
trec_covid = PromptTemplate(template=trec_covid_template, input_variables=["QUESTION"])
fiqa_template = """Please write a financial article passage to answer the question
Question: {QUESTION}
Passage:"""
fiqa = PromptTemplate(template=fiqa_template, input_variables=["QUESTION"])
dbpedia_entity_template = """Please write a passage to answer the question.
Question: {QUESTION}
Passage:"""
dbpedia_entity = PromptTemplate(
    template=dbpedia_entity_template, input_variables=["QUESTION"]
)
trec_news_template = """Please write a news passage about the topic.
Topic: {TOPIC}
Passage:"""
trec_news = PromptTemplate(template=trec_news_template, input_variables=["TOPIC"])
mr_tydi_template = """Please write a passage in Swahili/Korean/Japanese/Bengali to answer the question in detail.
Question: {QUESTION}
Passage:"""
mr_tydi = PromptTemplate(template=mr_tydi_template, input_variables=["QUESTION"])
PROMPT_MAP = {
    "web_search": web_search,
    "sci_fact": sci_fact,
    "arguana": arguana,
    "trec_covid": trec_covid,
    "fiqa": fiqa,
    "dbpedia_entity": dbpedia_entity,
    "trec_news": trec_news,
    "mr_tydi": mr_tydi,
}
