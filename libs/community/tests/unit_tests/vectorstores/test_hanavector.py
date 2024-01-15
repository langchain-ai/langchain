"""Test HanaVector functionality."""

from langchain_community.vectorstores.hanavector import HanaDB


def test_int_sanitation_with_illegal_value() -> None:
    """Test sanitization of int with illegal value"""
    successful = True
    try:
        HanaDB._sanitize_int("HUGO")
        successful = False
    except:
        pass
    
    assert successful

def test_int_sanitation_with_legal_values() -> None:
    """Test sanitization of int with legal values"""
    assert HanaDB._sanitize_int(42) == 42

    assert HanaDB._sanitize_int("21") == 21


def test_int_sanitation_with_negative_values() -> None:
    """Test sanitization of int with legal values"""
    assert HanaDB._sanitize_int(-1) == -1

    assert HanaDB._sanitize_int("-1") == -1

def test_int_sanitation_with_illegal_negative_value() -> None:
    """Test sanitization of int with illegal value"""
    successful = True
    try:
        HanaDB._sanitize_int(-2)
        successful = False
    except:
        pass
    
    assert successful

