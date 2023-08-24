import pathlib
from typing import Sequence

from langchain import schema
from langchain.chat_loaders import discord, utils


def _assert_messages_are_equal(
    actual_messages: Sequence[schema.BaseMessage],
    expected_messages: Sequence[schema.BaseMessage],
) -> None:
    assert len(actual_messages) == len(expected_messages)
    for actual, expected in zip(actual_messages, expected_messages):
        assert actual.content == expected.content
        assert (
            actual.additional_kwargs["sender"] == expected.additional_kwargs["sender"]
        )
        assert type(actual) == type(expected)


def test_discord_chat_loader() -> None:
    chat_path = pathlib.Path(__file__).parent / "data" / "discord_chat.txt"
    loader = discord.DiscordChatLoader(str(chat_path))

    chat_sessions = loader.load()
    chat_sessions = list(utils.map_ai_messages(chat_sessions, sender="reporterbob"))
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[0]["messages"], "Chat messages should not be empty"

    expected_content = [
        schema.HumanMessage(
            content="Hello Reporter Bob",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/11/2023 12:15 PM"}],
            },
        ),
        schema.AIMessage(
            content="Hi @talkingtower! I'm Bob, a journalist."
            " I'm writing a piece about famous landmarks.",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/11/2023 1:30 PM"}],
            },
        ),
        schema.HumanMessage(
            content="Interesting! But I'm just a tower. 😉",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/16/2023 1:42 PM"}],
            },
        ),
        schema.AIMessage(
            content="Oops! How about some virtual coffee?\n\nGot a fresh pot brewing.",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/11/2023 7:12 PM"}],
            },
        ),
        schema.HumanMessage(
            content="Of course! Virtual coffee sounds good.",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/11/2023 8:03 PM"}],
            },
        ),
        schema.AIMessage(
            content="Cheers! ☕️",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/14/2023 9:00 PM"}],
            },
        ),
        schema.HumanMessage(
            content="Thank you! Let's discuss the weather.",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/15/2023 7:43 AM"}],
            },
        ),
        schema.AIMessage(
            content="Sunny here! 🌞",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/15/2023 7:44 AM"}],
            },
        ),
        schema.HumanMessage(
            content="Rainy on my side! ☔️\nWebsite\nCheck today's weather"
            " forecast...\nWeather forecast in your area...",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/15/2023 8:30 AM"}],
            },
        ),
        schema.AIMessage(
            content="Stay dry! How about some music? 🎶"
            "\nWebsite\nTop 10 songs this week...",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/15/2023 9:56 AM"}],
            },
        ),
        schema.HumanMessage(
            content="Love music! Do you like jazz?",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "08/15/2023 11:10 AM"}],
            },
        ),
        schema.AIMessage(
            content="Yes! Jazz is fantastic. "
            "Ever heard this one?\nWebsite\nListen to classic jazz track...",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "08/15/2023 9:27 PM"}],
            },
        ),
        schema.HumanMessage(
            content="Indeed! Great choice. 🎷",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "Yesterday at 5:03 AM"}],
            },
        ),
        schema.AIMessage(
            content="Thanks! How about some virtual sightseeing?\nWebsite\n"
            "Virtual tour of famous landmarks...",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "Yesterday at 5:23 AM"}],
            },
        ),
        schema.HumanMessage(
            content="Sounds fun! Let's explore.",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "Today at 2:38 PM"}],
            },
        ),
        schema.AIMessage(
            content="Enjoy the tour! See you around.",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "Today at 2:56 PM"}],
            },
        ),
        schema.HumanMessage(
            content="Thank you! Goodbye! 👋",
            additional_kwargs={
                "sender": "talkingtower",
                "events": [{"message_time": "Today at 3:00 PM"}],
            },
        ),
        schema.AIMessage(
            content="Farewell! Happy exploring.",
            additional_kwargs={
                "sender": "reporterbob",
                "events": [{"message_time": "Today at 3:02 PM"}],
            },
        ),
    ]

    messages = chat_sessions[0]["messages"]
    _assert_messages_are_equal(messages, expected_content)
