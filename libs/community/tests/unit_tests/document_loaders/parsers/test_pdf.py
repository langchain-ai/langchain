from langchain_community.document_loaders.parsers.pdf import (
    _merge_text_and_extras,
)


def test_merge_text_and_extras() -> None:
    # assert ("abc\n\n\n<image>\n\n<table>\n\n\ndef"
    #     == _merge_text_and_extras(["<image>","<table>"],"abc\n\n\ndef"))
    # assert ("abc\n\n<image>\n\n<table>\n\ndef"
    #      ==   _merge_text_and_extras(["<image>","<table>"],"abc\n\ndef"))
    # assert ("abc\ndef\n\n<image>\n\n<table>"
    #         == _merge_text_and_extras(["<image>","<table>"],"abc\ndef"))

    assert "abc\n\n\n<image>\n\n<table>\n\n\ndef\n\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\n\ndef\n\n\nghi"
    )
    assert "abc\n\n<image>\n\n<table>\n\ndef\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\ndef\n\nghi"
    )
    assert "abc\ndef\n\n<image>\n\n<table>\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\ndef\n\nghi"
    )
