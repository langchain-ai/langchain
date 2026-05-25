import json

import pytest

from langchain_core.load import dump, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate


def test_image_prompt_template_deserializable() -> None:
    """Test that the image prompt template is serializable."""
    loads(
        dump.dumps(
            ChatPromptTemplate.from_messages(
                [("system", [{"type": "image", "image_url": "{img}"}])]
            )
        ),
    )


def test_image_prompt_template_deserializable_old() -> None:
    """Test that the image prompt template is serializable."""
    loads(
        json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
                "kwargs": {
                    "messages": [
                        {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "langchain",
                                "prompts",
                                "chat",
                                "SystemMessagePromptTemplate",
                            ],
                            "kwargs": {
                                "prompt": [
                                    {
                                        "lc": 1,
                                        "type": "constructor",
                                        "id": [
                                            "langchain",
                                            "prompts",
                                            "prompt",
                                            "PromptTemplate",
                                        ],
                                        "kwargs": {
                                            "template": "Foo",
                                            "input_variables": [],
                                            "template_format": "f-string",
                                            "partial_variables": {},
                                        },
                                    }
                                ]
                            },
                        },
                        {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "langchain",
                                "prompts",
                                "chat",
                                "HumanMessagePromptTemplate",
                            ],
                            "kwargs": {
                                "prompt": [
                                    {
                                        "lc": 1,
                                        "type": "constructor",
                                        "id": [
                                            "langchain",
                                            "prompts",
                                            "image",
                                            "ImagePromptTemplate",
                                        ],
                                        "kwargs": {
                                            "template": {
                                                "url": "data:image/png;base64,{img}"
                                            },
                                            "input_variables": ["img"],
                                        },
                                    },
                                    {
                                        "lc": 1,
                                        "type": "constructor",
                                        "id": [
                                            "langchain",
                                            "prompts",
                                            "prompt",
                                            "PromptTemplate",
                                        ],
                                        "kwargs": {
                                            "template": "{input}",
                                            "input_variables": ["input"],
                                            "template_format": "f-string",
                                            "partial_variables": {},
                                        },
                                    },
                                ]
                            },
                        },
                    ],
                    "input_variables": ["img", "input"],
                },
            }
        ),
    )


def test_image_prompt_template_rejects_attribute_access_in_template_values() -> None:
    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        ImagePromptTemplate(
            input_variables=["image"],
            template={"url": "https://example.com/{image.__class__.__name__}.png"},
        )


def test_image_prompt_template_deserialization_rejects_attribute_access() -> None:
    payload = json.dumps(
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "image", "ImagePromptTemplate"],
            "kwargs": {
                "template": {
                    "url": "https://example.com/{image.__class__.__name__}.png"
                },
                "input_variables": ["image"],
                "template_format": "f-string",
            },
        }
    )

    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        loads(payload)
