from langchain_core.load import dumpd
from langchain_core.messages import HumanMessage


def test_pre_0_1_30_message_compat() -> None:
    msg = HumanMessage("foo", data={"bar": 1})
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
    actual = dumpd(msg)
    _is_superset(dump_0_1_29, actual)


def _is_superset(expected: dict, actual: dict) -> None:
    for k, v in expected.items():
        if isinstance(v, dict):
            _is_superset(v, actual[k])
        else:
            assert actual[k] == v
