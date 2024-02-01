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
                "to": "+1234567890",
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
            to="+1234567890",
            sender="+1234567890",
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
                "to": "test@test.com",
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