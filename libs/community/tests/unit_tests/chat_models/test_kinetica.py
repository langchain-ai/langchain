"""Test `Kinetica` chat models"""

import logging
from typing import Any

from langchain_core.messages import AIMessage

from langchain_community.chat_models.kinetica import ChatKinetica, KineticaUtil

LOG = logging.getLogger(__name__)


class TestChatKinetica:
    test_ctx_json: str = """
    {
        "payload":{
            "context":[
                {
                    "table":"demo.test_profiles",
                    "columns":[
                    "username VARCHAR (32) NOT NULL",
                    "name VARCHAR (32) NOT NULL",
                    "sex VARCHAR (1) NOT NULL",
                    "address VARCHAR (64) NOT NULL",
                    "mail VARCHAR (32) NOT NULL",
                    "birthdate TIMESTAMP NOT NULL"
                    ],
                    "description":"Contains user profiles.",
                    "rules":[
                    
                    ]
                },
                {
                    "samples":{
                    "How many male users are there?":
    "select count(1) as num_users from demo.test_profiles where sex = ''M'';"
                    }
                }
            ]
        }
    }
"""

    def test_convert_messages(self, monkeypatch: Any) -> None:
        """Test convert messages from context."""

        def patch_kdbc() -> None:
            return None

        monkeypatch.setattr(KineticaUtil, "create_kdbc", patch_kdbc)

        def patch_execute_sql(*args: Any, **kwargs: Any) -> dict:
            return dict(Prompt=self.test_ctx_json)

        monkeypatch.setattr(ChatKinetica, "_execute_sql", patch_execute_sql)

        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]

        test_messages = kinetica_llm.load_messages_from_context("test")
        LOG.info(f"test_messages: {test_messages}")
        ai_message = test_messages[-1]
        assert isinstance(ai_message, AIMessage)
        assert (
            ai_message.content
            == "select count(1) as num_users from demo.test_profiles where sex = 'M';"
        )
