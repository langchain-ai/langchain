import pytest

from langchain_core.utils.function_calling import _rm_titles

output1 = {
    "type": "object",
    "properties": {
        "people": {
            "description": "List of info about people",
            "type": "array",
            "items": {
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"description": "person's age", "type": "integer"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["people"],
}

schema1 = {
    "type": "object",
    "properties": {
        "people": {
            "title": "People",
            "description": "List of info about people",
            "type": "array",
            "items": {
                "title": "Person",
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "title": {
                        "title": "Title",
                        "description": "person's age",
                        "type": "integer",
                    },
                },
                "required": ["name"],
            },
        }
    },
    "required": ["people"],
}

output2 = {
    "type": "object",
    "properties": {
        "title": {
            "description": "List of info about people",
            "type": "array",
            "items": {
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"description": "person's age", "type": "integer"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["title"],
}

schema2 = {
    "type": "object",
    "properties": {
        "title": {
            "title": "Title",
            "description": "List of info about people",
            "type": "array",
            "items": {
                "title": "Person",
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "age": {
                        "title": "Age",
                        "description": "person's age",
                        "type": "integer",
                    },
                },
                "required": ["name"],
            },
        }
    },
    "required": ["title"],
}


output3 = {
    "type": "object",
    "properties": {
        "title": {
            "description": "List of info about people",
            "type": "array",
            "items": {
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {"description": "person's age", "type": "integer"},
                },
                "required": ["title"],
            },
        }
    },
    "required": ["title"],
}

schema3 = {
    "type": "object",
    "properties": {
        "title": {
            "title": "Title",
            "description": "List of info about people",
            "type": "array",
            "items": {
                "title": "Person",
                "description": "Information about a person.",
                "type": "object",
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "type": {
                        "title": "Type",
                        "description": "person's age",
                        "type": "integer",
                    },
                },
                "required": ["title"],
            },
        }
    },
    "required": ["title"],
}


output4 = {
    "type": "object",
    "properties": {
        "properties": {
            "description": "Information to extract",
            "type": "object",
            "properties": {
                "title": {
                    "description": "Information about papers mentioned.",
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                    },
                    "required": ["title"],
                }
            },
            "required": ["title"],
        }
    },
    "required": ["properties"],
}

schema4 = {
    "type": "object",
    "properties": {
        "properties": {
            "title": "Info",
            "description": "Information to extract",
            "type": "object",
            "properties": {
                "title": {
                    "title": "Paper",
                    "description": "Information about papers mentioned.",
                    "type": "object",
                    "properties": {
                        "title": {"title": "Title", "type": "string"},
                        "author": {"title": "Author", "type": "string"},
                    },
                    "required": ["title"],
                }
            },
            "required": ["title"],
        }
    },
    "required": ["properties"],
}

schema5 = {
    "description": "A list of data.",
    "items": {
        "description": "foo",
        "properties": {
            "title": {"type": "string", "description": "item title"},
            "due_date": {"type": "string", "description": "item due date"},
        },
        "required": [],
        "type": "object",
    },
    "type": "array",
}

output5 = {
    "description": "A list of data.",
    "items": {
        "description": "foo",
        "properties": {
            "title": {"type": "string", "description": "item title"},
            "due_date": {"type": "string", "description": "item due date"},
        },
        "required": [],
        "type": "object",
    },
    "type": "array",
}


@pytest.mark.parametrize(
    ("schema", "output"),
    [
        (schema1, output1),
        (schema2, output2),
        (schema3, output3),
        (schema4, output4),
        (schema5, output5),
    ],
)
def test_rm_titles(schema: dict, output: dict) -> None:
    assert _rm_titles(schema) == output
