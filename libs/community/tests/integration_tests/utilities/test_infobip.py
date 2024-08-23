from typing import Dict

import responses

from langchain_community.utilities.infobip import InfobipAPIWrapper


def test_send_sms() -> None:
    infobip: InfobipAPIWrapper = InfobipAPIWrapper(
        infobip_api_key="test",
        infobip_base_url="https://api.infobip.com",
    )

    json_response: Dict = {
        "messages": [
            {
                "messageId": "123",
                "status": {
                    "description": "Message sent to next instance",
                    "groupId": 1,
                    "groupName": "PENDING",
                    "id": 26,
                    "name": "PENDING_ACCEPTED",
                },
                "to": "41793026727",
            }
        ]
    }

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://api.infobip.com/sms/2/text/advanced",
            json=json_response,
            status=200,
        )

        response: str = infobip.run(
            body="test",
            to="41793026727",
            sender="41793026727",
            channel="sms",
        )
        assert response == "123"


def test_send_email() -> None:
    infobip: InfobipAPIWrapper = InfobipAPIWrapper(
        infobip_api_key="test",
        infobip_base_url="https://api.infobip.com",
    )

    json_response: Dict = {
        "bulkId": "123",
        "messages": [
            {
                "to": "test@example.com",
                "messageId": "123",
                "status": {
                    "groupId": 1,
                    "groupName": "PENDING",
                    "id": 26,
                    "name": "PENDING_ACCEPTED",
                    "description": "Message accepted, pending for delivery.",
                },
            }
        ],
    }

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://api.infobip.com/email/3/send",
            json=json_response,
            status=200,
        )

        response: str = infobip.run(
            body="test",
            to="test@example.com",
            sender="test@example.com",
            subject="test",
            channel="email",
        )

        assert response == "123"
