import json
import os
import uuid
from typing import Generator, Union

import pytest

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ElasticsearchChatMessageHistory
from langchain.schema.messages import _message_to_dict

"""
cd tests/integration_tests/memory/docker-compose
docker-compose -f elasticsearch.yml up

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_USERNAME
- ES_PASSWORD
"""


class TestElasticsearch:
    @pytest.fixture(scope="class", autouse=True)
    def elasticsearch_connection(self) -> Union[dict, Generator[dict, None, None]]:
        # Run this integration test against Elasticsearch on localhost,
        # or an Elastic Cloud instance
        from elasticsearch import Elasticsearch

        es_url = os.environ.get("ES_URL", "http://localhost:9200")
        es_cloud_id = os.environ.get("ES_CLOUD_ID")
        es_username = os.environ.get("ES_USERNAME", "elastic")
        es_password = os.environ.get("ES_PASSWORD", "changeme")

        if es_cloud_id:
            es = Elasticsearch(
                cloud_id=es_cloud_id,
                basic_auth=(es_username, es_password),
            )
            yield {
                "es_cloud_id": es_cloud_id,
                "es_user": es_username,
                "es_password": es_password,
            }

        else:
            # Running this integration test with local docker instance
            es = Elasticsearch(hosts=es_url)
            yield {"es_url": es_url}

        # Clear all indexes
        index_names = es.indices.get(index="_all").keys()
        for index_name in index_names:
            if index_name.startswith("test_"):
                es.indices.delete(index=index_name)
        es.indices.refresh(index="_all")

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    def test_memory_with_message_store(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test the memory with a message store."""
        # setup Elasticsearch as a message store
        message_history = ElasticsearchChatMessageHistory(
            **elasticsearch_connection, index=index_name, session_id="test-session"
        )

        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )

        # add some messages
        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")

        # get the message history from the memory store and turn it into a json
        messages = memory.chat_memory.messages
        messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

        assert "This is me, the AI" in messages_json
        assert "This is me, the human" in messages_json

        # remove the record from Elasticsearch, so the next test run won't pick it up
        memory.chat_memory.clear()

        assert memory.chat_memory.messages == []
