from langchain_core.load.load import load
from langchain_core.messages import HumanMessage


def test_pre_0_1_30_message_compat() -> None:
    expected = HumanMessage("foo", data={"bar": 1})
    # Old message serialization that only has 'additional_kwargs' and not 'data'
    dump_0_1_29 = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "HumanMessage"],
        "kwargs": {
            "content": "foo",
            "additional_kwargs": {"bar": 1},
        },
    }
    actual = load(dump_0_1_29)
    assert actual == expected
