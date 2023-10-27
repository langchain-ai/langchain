import os

from elasticsearch import Elasticsearch

es_host = os.environ["ELASTIC_SEARCH_SERVER"]
es_password = os.environ["ELASTIC_PASSWORD"]

db = Elasticsearch(
    es_host,
    http_auth=('elastic', es_password),
    ca_certs='http_ca.crt'  # Replace with your actual path
)

customers = [
    {"firstname": "Jennifer", "lastname": "Walters"},
    {"firstname": "Monica","lastname":"Rambeau"},
    {"firstname": "Carol","lastname":"Danvers"},
    {"firstname": "Wanda","lastname":"Maximoff"},
    {"firstname": "Jennifer","lastname":"Takeda"},
]
for i, customer in enumerate(customers):
    db.create(index="customers", document=customer, id=i)
