import pytest

from langchain.agents.protocol import (
    BackendProtocol,
    GrepResult,
    ReadResult,
    _apply_grep_max_count,
    _supports_delete,
)


class TestReadResult:
    def test_valid_read_result(self) -> None:
        result = ReadResult(
            file_data={"content": "abc", "encoding": "utf-8"},
            start_line=1,
            end_line=5,
            total_lines=10,
            next_offset=5,
        )

        assert result.start_line == 1
        assert result.end_line == 5
        assert result.total_lines == 10
        assert result.next_offset == 5

    def test_start_without_end_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="start_line and end_line must be set together",
        ):
            ReadResult(start_line=1)

    def test_end_without_start_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="start_line and end_line must be set together",
        ):
            ReadResult(end_line=10)

    def test_next_offset_requires_window(self) -> None:
        with pytest.raises(
            ValueError,
            match="next_offset requires start_line",
        ):
            ReadResult(next_offset=10)

    def test_total_lines_requires_window(self) -> None:
        with pytest.raises(
            ValueError,
            match="total_lines requires start_line",
        ):
            ReadResult(total_lines=100)

    def test_start_line_must_be_positive(self) -> None:
        with pytest.raises(
            ValueError,
            match="window must satisfy",
        ):
            ReadResult(
                start_line=0,
                end_line=1,
            )

    def test_end_before_start(self) -> None:
        with pytest.raises(
            ValueError,
            match="window must satisfy",
        ):
            ReadResult(
                start_line=5,
                end_line=4,
            )

    def test_total_lines_less_than_end_line(self) -> None:
        with pytest.raises(
            ValueError,
            match="cannot be less than end_line",
        ):
            ReadResult(
                start_line=1,
                end_line=10,
                total_lines=9,
            )

    def test_next_offset_must_equal_end_line(self) -> None:
        with pytest.raises(
            ValueError,
            match="must equal end_line",
        ):
            ReadResult(
                start_line=1,
                end_line=10,
                next_offset=9,
            )

class TestApplyGrepMaxCount:
    def test_no_limit(self) -> None:
        result = GrepResult(
            matches=[{"line": 1}, {"line": 2}],
            truncated=False,
        )

        new_result = _apply_grep_max_count(result, None)

        assert new_result is result

    def test_matches_less_than_limit(self) -> None:
        result = GrepResult(
            matches=[{"line": 1}, {"line": 2}],
            truncated=False,
        )

        new_result = _apply_grep_max_count(result, 5)

        assert new_result is result

    def test_matches_equal_limit(self) -> None:
        result = GrepResult(
            matches=[{"line": 1}, {"line": 2}],
            truncated=False,
        )

        new_result = _apply_grep_max_count(result, 2)

        assert new_result is result

    def test_matches_greater_than_limit(self) -> None:
        result = GrepResult(
            matches=[{"line": 1}, {"line": 2}, {"line": 3}],
            truncated=False,
        )

        new_result = _apply_grep_max_count(result, 2)

        assert len(new_result.matches) == 2
        assert new_result.truncated is True

class BackendWithoutDelete(BackendProtocol):
    pass

class BackendWithDelete(BackendProtocol):
    def delete() -> None:
        return None

def test_supports_delete_false() -> None:
    assert _supports_delete(BackendWithoutDelete()) is False

def test_supports_delete_true() -> None:
    assert _supports_delete(BackendWithDelete()) is True
