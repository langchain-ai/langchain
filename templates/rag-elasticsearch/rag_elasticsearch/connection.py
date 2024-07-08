import os

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")

if ELASTIC_CLOUD_ID and ELASTIC_USERNAME and ELASTIC_PASSWORD:
    es_connection_details = {
        "es_cloud_id": ELASTIC_CLOUD_ID,
        "es_user": ELASTIC_USERNAME,
        "es_password": ELASTIC_PASSWORD,
    }
else:
    es_connection_details = {"es_url": ES_URL}
