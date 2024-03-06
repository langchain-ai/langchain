from typing import Any, Dict, Optional

from elasticsearch import Elasticsearch


def create_elasticsearch_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Elasticsearch:
    if url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if url:
        connection_params["hosts"] = [url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    if params is not None:
        connection_params.update(params)

    es_client = Elasticsearch(**connection_params)

    es_client.info()  # test connection

    return es_client
