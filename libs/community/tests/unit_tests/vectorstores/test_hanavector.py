"""Test HanaVector functionality."""

from langchain_community.vectorstores.hanavector import HanaDB


def test_int_sanitation_with_illegal_value() -> None:
    """Test sanitization of int with illegal value"""
    try:
        HanaDB._sanitize_int("HUGO")
        assert False
    except:
        pass

def test_int_sanitation_with_legal_values() -> None:
    """Test sanitization of int with legal values"""
    assert HanaDB._sanitize_int(42) == 42

    assert HanaDB._sanitize_int("21") == 21
