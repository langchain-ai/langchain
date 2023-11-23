# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

web_search_template = """Пожалуйста, напиши текст, чтобы ответить на вопрос 
Question: {QUESTION}
Текст:"""
web_search = PromptTemplate(template=web_search_template, input_variables=["QUESTION"])
sci_fact_template = """Пожалуйста, напиши отрывок из научной статьи, чтобы подтвердить/опровергнуть утверждение 
Утверждение: {Claim}
Текст:"""
sci_fact = PromptTemplate(template=sci_fact_template, input_variables=["Claim"])
arguana_template = """Пожалуйста, напиши контраргумент к тексту 
Текст: {PASSAGE}
Контраргумент:"""
arguana = PromptTemplate(template=arguana_template, input_variables=["PASSAGE"])
trec_covid_template = """Пожалуйста, напиши отрывок из научной статьи, чтобы ответить на вопрос
Question: {QUESTION}
Текст:"""
trec_covid = PromptTemplate(template=trec_covid_template, input_variables=["QUESTION"])
fiqa_template = """Пожалуйста, напиши отрывок из финансовой статьи, чтобы ответить на вопрос
Question: {QUESTION}
Текст:"""
fiqa = PromptTemplate(template=fiqa_template, input_variables=["QUESTION"])
dbpedia_entity_template = """Пожалуйста, напиши текст, чтобы ответить на вопрос.
Question: {QUESTION}
Текст:"""
dbpedia_entity = PromptTemplate(
    template=dbpedia_entity_template, input_variables=["QUESTION"]
)
trec_news_template = """Пожалуйста, напиши новостной отрывок на заданную тему.
Тема: {TOPIC}
Текст:"""
trec_news = PromptTemplate(template=trec_news_template, input_variables=["TOPIC"])
mr_tydi_template = """Пожалуйста, напиши текст на свахили/корейском/японском/бенгальском, чтобы подробно ответить на вопрос.
Question: {QUESTION}
Текст:"""
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
