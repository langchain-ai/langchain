import pytest

from langchain.agents.utils import (
    compile_grep_include_glob,
    create_file_data,
    format_content_with_line_numbers,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    validate_path,
)


@pytest.mark.requires("wcmatch")
def test_validate_path_normalizes_relative_path() -> None:
    assert validate_path("foo/bar") == "/foo/bar"

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_parent_traversal() -> None:
    with pytest.raises(
        ValueError,
        match=r"Path traversal not allowed",
    ):
        validate_path("../etc/passwd")

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_windows_absolute_path() -> None:
    with pytest.raises(
        ValueError,
        match=r"Windows absolute paths are not supported",
    ):
        validate_path(r"C:\Users\test.txt")

@pytest.mark.requires("wcmatch")
def test_format_content_with_line_numbers() -> None:
    result = format_content_with_line_numbers(
        "hello\nworld"
    )

    assert result == (
        "1  hello\n"
        "2  world"
    )

@pytest.mark.requires("wcmatch")
def test_format_long_line() -> None:
    content = "a" * 6000

    result = format_content_with_line_numbers(content)

    assert "1.1" in result

@pytest.mark.requires("wcmatch")
def test_slice_read_response() -> None:
    file_data = {
        "content": "a\nb\nc",
        "encoding": "utf-8"
    }

    result = slice_read_response(
        file_data,
        offset=1,
        limit=1
    )

    assert result.file_data["content"] == "b\n"
    assert result.start_line == 2
    assert result.end_line == 2

@pytest.mark.requires("wcmatch")
def test_slice_read_response_invalid_offset() -> None:
    result = slice_read_response(
        {
            "content": "hello",
            "encoding": "utf-8"
        },
        offset=10,
        limit=2
    )

    assert result.error is not None

@pytest.mark.requires("wcmatch")
def test_replace_string() -> None:
    result = perform_string_replacement(
        "hello world",
        "world",
        "python"
    )

    assert result == ("hello python", 1)

@pytest.mark.requires("wcmatch")
def test_replace_missing_string() -> None:

    result = perform_string_replacement(
        "abc",
        "xyz",
        "123",
    )

    assert isinstance(result, str)
    assert "not found" in result

@pytest.mark.requires("wcmatch")
def test_replace_duplicate_without_replace_all() -> None:
    result = perform_string_replacement(
        "a a",
        "a",
        "b"
    )

    assert isinstance(result, str)
    assert "replace_all=True" in result

@pytest.mark.requires("wcmatch")
def test_compile_grep_include_glob_matches_python_files() -> None:
    matcher = compile_grep_include_glob("*.py")

    assert matcher("src/main.py")

@pytest.mark.requires("wcmatch")
def test_compile_grep_include_glob_rejects_txt() -> None:
    matcher = compile_grep_include_glob("*.py")

    assert not matcher("src/main.txt")

@pytest.mark.requires("wcmatch")
def test_grep_matches_files() -> None:

    files = {
        "/a.py": {
            "content": "hello\nworld",
            "encoding": "utf-8"
        }
    }

    result = grep_matches_from_files(
        files,
        "hello"
    )

    assert result.matches == [
        {
            "path": "/a.py",
            "line": 1,
            "text": "hello"
        }
    ]

@pytest.mark.requires("wcmatch")
def test_create_file_data() -> None:
    result = create_file_data("hello")

    assert result["content"] == "hello"
    assert result["encoding"] == "utf-8"
    assert "created_at" in result

@pytest.mark.requires("wcmatch")
def test_validate_path_root() -> None:
    assert validate_path("/") == "/"

@pytest.mark.requires("wcmatch")
def test_validate_path_removes_duplicate_slashes() -> None:
    assert validate_path("/foo//bar") == "/foo/bar"

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_parent_directory_component() -> None:
    with pytest.raises(ValueError, match="Path traversal"):
        validate_path("/foo/../bar")

@pytest.mark.requires("wcmatch")
def test_validate_path_allows_filename_with_dots() -> None:
    assert validate_path("/foo..bar.txt") == "/foo..bar.txt"

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_home_expansion() -> None:
    with pytest.raises(
        ValueError,
        match=r"Path traversal not allowed",
    ):
        validate_path("~/secret.txt")

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_windows_drive_path() -> None:
    with pytest.raises(ValueError, match="Windows absolute paths"):
        validate_path(r"C:\Users\test.txt")

@pytest.mark.requires("wcmatch")
def test_validate_path_checks_allowed_prefixes() -> None:
    assert validate_path(
        "/data/file.txt",
        allowed_prefixes=["/data"],
    ) == "/data/file.txt"

@pytest.mark.requires("wcmatch")
def test_validate_path_rejects_disallowed_prefix() -> None:
    with pytest.raises(
        ValueError,
        match=r"Path must start with one of",
    ):
        validate_path(
            "/etc/passwd",
            allowed_prefixes=["/data"],
        )

@pytest.mark.requires("wcmatch")
def test_format_empty_content() -> None:
    result = format_content_with_line_numbers("")

    assert result == ""

@pytest.mark.requires("wcmatch")
def test_format_content_with_trailing_newline() -> None:
    result = format_content_with_line_numbers("hello\n")

    assert result == "1  hello"

@pytest.mark.requires("wcmatch")
def test_format_content_custom_start_line() -> None:
    result = format_content_with_line_numbers(
        "hello",
        start_line=10,
    )

    assert result == "10  hello"

@pytest.mark.requires("wcmatch")
def test_format_long_line_creates_continuation_marker() -> None:
    content = "a" * 6000

    result = format_content_with_line_numbers(content)

    assert "1.1" in result

@pytest.mark.requires("wcmatch")
def make_file(content: str) -> None:
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": "today",
    }

@pytest.mark.requires("wcmatch")
def test_slice_empty_file() -> None:
    result = slice_read_response(
        make_file(""),
        offset=0,
        limit=10,
    )

    assert result.file_data["content"] == ""
    assert result.start_line is None

@pytest.mark.requires("wcmatch")
def test_slice_offset_beyond_file() -> None:

    result = slice_read_response(
        make_file("a\nb"),
        offset=10,
        limit=5,
    )

    assert result.error is not None

@pytest.mark.requires("wcmatch")
def test_slice_preserves_metadata() -> None:
    result = slice_read_response(
        make_file("hello"),
        offset=0,
        limit=1,
    )

    assert result.file_data["created_at"] == "today"

@pytest.mark.requires("wcmatch")
def test_slice_handles_crlf() -> None:

    result = slice_read_response(
        make_file("a\r\nb\r\nc"),
        offset=0,
        limit=3,
    )

    assert result.file_data["content"] == "a\nb\nc"

@pytest.mark.requires("wcmatch")
def test_slice_returns_next_offset() -> None:

    result = slice_read_response(
        make_file("a\nb\nc"),
        offset=0,
        limit=2,
    )

    assert result.next_offset == 2

@pytest.mark.requires("wcmatch")
def test_replace_single_occurrence() -> None:
    result = perform_string_replacement(
        "hello world",
        "world",
        "python",
    )

    assert result == ("hello python", 1)

@pytest.mark.requires("wcmatch")
def test_replace_all_occurrences() -> None:
    result = perform_string_replacement(
        "a a a",
        "a",
        "b",
        replace_all=True,
    )

    assert result == ("b b b", 3)

@pytest.mark.requires("wcmatch")
def test_replace_missing_final_newline_hint() -> None:

    result = perform_string_replacement(
        "hello",
        "hello\n",
        "bye\n",
    )

    assert isinstance(result, str)
    assert "trailing newline" in result

@pytest.mark.requires("wcmatch")
def test_grep_finds_literal_match() -> None:

    files = {
        "/a.py": {
            "content": "hello\nworld",
            "encoding": "utf-8",
        }
    }

    result = grep_matches_from_files(
        files,
        "hello",
    )

    assert result.matches[0]["line"] == 1

@pytest.mark.requires("wcmatch")
def test_grep_is_literal_not_regex() -> None:

    files = {
        "/a.py": {
            "content": "a.*b",
            "encoding": "utf-8",
        }
    }

    result = grep_matches_from_files(
        files,
        ".*",
    )

    assert len(result.matches) == 1

@pytest.mark.requires("wcmatch")
def test_grep_respects_max_count() -> None:

    files = {
        "/a.py": {
            "content": "x\nx\nx",
            "encoding": "utf-8",
        }
    }

    result = grep_matches_from_files(
        files,
        "x",
        max_count=2,
    )

    assert len(result.matches) == 2
    assert result.truncated is True

@pytest.mark.requires("wcmatch")
def test_grep_invalid_path_returns_empty() -> None:

    result = grep_matches_from_files(
        {},
        "test",
        path="",
    )

    assert result.matches == []

@pytest.mark.requires("wcmatch")
def test_glob_matches_basename() -> None:

    matcher = compile_grep_include_glob("*.py")

    assert matcher("src/main.py")

@pytest.mark.requires("wcmatch")
def test_glob_does_not_match_wrong_extension() -> None:

    matcher = compile_grep_include_glob("*.py")

    assert not matcher("src/main.txt")

@pytest.mark.requires("wcmatch")
def test_glob_recursive_pattern() -> None:

    matcher = compile_grep_include_glob("src/**/*.py")

    assert matcher("src/app/main.py")
