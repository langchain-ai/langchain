import os
from pathlib import Path

from elasticsearch import Elasticsearch
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

from .elastic_index_info import get_indices_infos
from .prompts import DSL_PROMPT

es_host = os.environ["ELASTIC_SEARCH_SERVER"]
es_password = os.environ["ELASTIC_PASSWORD"]

db = Elasticsearch(
    es_host,
    http_auth=('elastic', es_password),
    ca_certs=Path(__file__).parents[1] / 'http_ca.crt'  # Replace with your actual path
)

_model = ChatOpenAI(temperature=0, model="gpt-4")

chain = {
    "input": lambda x: x["input"],
    "indices_info": lambda _: get_indices_infos(db),
    "top_k": lambda x: x.get("top_k", 5),
} | DSL_PROMPT | _model | SimpleJsonOutputParser()
