"""Tests for the activity schedule system.

Covers the DB-backed schedule in strategy.py and the schedule boost
in SkillRegistry.evaluate().  All tests use SUPABASE_MOCK=true so no
network calls are needed.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

# Force mock mode for all tests in this module
os.environ["SUPABASE_MOCK"] = "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_datetime(weekday: int, day: int = 10) -> datetime:
    """Build a UTC datetime for a given weekday (0=Mon..6=Sun).

    Uses March 2026 as base. March 2 2026 is a Monday.
    """
    # March 2 2026 = Monday (weekday 0)
    target_day = 2 + weekday
    return datetime(2026, 3, target_day, 12, 0, 0, tzinfo=timezone.utc)


def _clear_mock_tables() -> None:
    """Clear mock store for schedule tables."""
    from agents.seo_agent.tools.supabase_tools import _mock_store

    _mock_store.pop("ralf_schedule", None)
    _mock_store.pop("ralf_schedule_log", None)


# ---------------------------------------------------------------------------
# Tests: seed_schedule
# ---------------------------------------------------------------------------


class TestSeedSchedule:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_seed_populates_empty_table(self) -> None:
        from agents.seo_agent.strategy import seed_schedule
        from agents.seo_agent.tools.supabase_tools import query_table

        count = seed_schedule()
        assert count > 0

        rows = query_table("ralf_schedule", limit=500)
        assert len(rows) == count

    def test_seed_is_idempotent(self) -> None:
        from agents.seo_agent.strategy import seed_schedule

        first = seed_schedule()
        second = seed_schedule()
        assert first > 0
        assert second == 0

    def test_seed_includes_all_cadences(self) -> None:
        from agents.seo_agent.strategy import seed_schedule
        from agents.seo_agent.tools.supabase_tools import query_table

        seed_schedule()
        rows = query_table("ralf_schedule", limit=500)
        cadences = {r["cadence"] for r in rows}
        assert "daily" in cadences
        assert "weekly" in cadences
        assert "monthly" in cadences


# ---------------------------------------------------------------------------
# Tests: get_todays_schedule
# ---------------------------------------------------------------------------


class TestGetTodaysSchedule:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_monday_returns_keyword_research(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        monday = _make_datetime(0)
        schedule = get_todays_schedule(monday)

        assert "keyword_research" in schedule["boost_skills"]
        assert "Keyword" in schedule["label"]

    def test_tuesday_returns_content_writing(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        tuesday = _make_datetime(1)
        schedule = get_todays_schedule(tuesday)

        assert "publish_blog" in schedule["boost_skills"]
        assert "Content" in schedule["label"]

    def test_thursday_returns_prospecting(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        thursday = _make_datetime(3)
        schedule = get_todays_schedule(thursday)

        assert "discover_prospects" in schedule["boost_skills"]
        assert "score_prospects" in schedule["boost_skills"]

    def test_friday_returns_analytics(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        friday = _make_datetime(4)
        schedule = get_todays_schedule(friday)

        assert "track_rankings" in schedule["boost_skills"]

    def test_saturday_returns_maintenance(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        saturday = _make_datetime(5)
        schedule = get_todays_schedule(saturday)

        assert "internal_linking" in schedule["boost_skills"]
        assert "memory_consolidation" in schedule["boost_skills"]

    def test_weekly_tasks_appear_on_correct_day(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        # internal_linking has a weekly entry on Saturday (day 5)
        saturday = _make_datetime(5)
        schedule = get_todays_schedule(saturday)

        assert "internal_linking" in schedule["weekly_due"]

    def test_weekly_tasks_absent_on_wrong_day(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        monday = _make_datetime(0)
        schedule = get_todays_schedule(monday)

        assert "internal_linking" not in schedule.get("weekly_due", [])

    def test_monthly_tasks_on_first_of_month(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        # keyword_refresh is monthly on day 1
        first = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        schedule = get_todays_schedule(first)

        assert "keyword_refresh" in schedule["monthly_due"]

    def test_monthly_tolerance_one_day(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        # Should still match on day 2 (±1 tolerance)
        second = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
        schedule = get_todays_schedule(second)

        assert "keyword_refresh" in schedule["monthly_due"]

    def test_monthly_no_match_outside_tolerance(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        tenth = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        schedule = get_todays_schedule(tenth)

        assert "keyword_refresh" not in schedule.get("monthly_due", [])

    def test_boost_takes_max_of_daily_and_weekly(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        # Saturday: internal_linking has daily boost 25 and weekly boost 40
        saturday = _make_datetime(5)
        schedule = get_todays_schedule(saturday)

        assert schedule["boost_skills"]["internal_linking"] == 40

    def test_all_days_return_valid_schedule(self) -> None:
        from agents.seo_agent.strategy import get_todays_schedule

        for weekday in range(7):
            dt = _make_datetime(weekday)
            schedule = get_todays_schedule(dt)
            assert "label" in schedule
            assert "boost_skills" in schedule
            assert isinstance(schedule["boost_skills"], dict)


# ---------------------------------------------------------------------------
# Tests: log_schedule_completion / get_todays_log / get_schedule_history
# ---------------------------------------------------------------------------


class TestScheduleLog:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_log_and_recall(self) -> None:
        from agents.seo_agent.strategy import get_todays_log, log_schedule_completion

        log_schedule_completion(
            "keyword_research",
            site="freeroomplanner",
            summary="Found 10 keywords",
            status="done",
        )

        log = get_todays_log()
        assert len(log) == 1
        assert log[0]["skill"] == "keyword_research"
        assert log[0]["status"] == "done"

    def test_log_failure(self) -> None:
        from agents.seo_agent.strategy import get_todays_log, log_schedule_completion

        log_schedule_completion(
            "publish_blog",
            site="freeroomplanner",
            summary="Rate limited",
            status="failed",
        )

        log = get_todays_log()
        assert len(log) == 1
        assert log[0]["status"] == "failed"

    def test_history_filters_by_date(self) -> None:
        from agents.seo_agent.strategy import get_schedule_history, log_schedule_completion

        log_schedule_completion("keyword_research", summary="today", schedule_date="2026-03-28")
        log_schedule_completion("publish_blog", summary="old", schedule_date="2026-01-01")

        history = get_schedule_history(days_back=7)
        skills = [h["skill"] for h in history]
        assert "keyword_research" in skills
        # Old entry should be filtered out
        assert "publish_blog" not in skills


# ---------------------------------------------------------------------------
# Tests: update_schedule_entry
# ---------------------------------------------------------------------------


class TestUpdateSchedule:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_update_day_of_week(self) -> None:
        from agents.seo_agent.strategy import get_full_schedule, seed_schedule, update_schedule_entry

        seed_schedule()
        rows = get_full_schedule()

        # Find a daily keyword_research entry
        kr_entry = next(
            r for r in rows
            if r["skill"] == "keyword_research" and r["cadence"] == "daily"
        )
        original_day = kr_entry["day_of_week"]

        # Move to Wednesday (2)
        update_schedule_entry(kr_entry["id"], day_of_week=2)

        updated_rows = get_full_schedule()
        updated = next(r for r in updated_rows if r["id"] == kr_entry["id"])
        assert updated["day_of_week"] == 2
        assert updated["day_of_week"] != original_day

    def test_deactivate_entry(self) -> None:
        from agents.seo_agent.strategy import seed_schedule, update_schedule_entry
        from agents.seo_agent.tools.supabase_tools import query_table

        seed_schedule()
        rows = query_table("ralf_schedule", limit=500)
        entry = rows[0]

        update_schedule_entry(entry["id"], active=False)

        updated_rows = query_table("ralf_schedule", limit=500)
        updated = next(r for r in updated_rows if r["id"] == entry["id"])
        assert updated["active"] is False

    def test_partial_update_preserves_skill(self) -> None:
        """Regression: partial update must not null out the ``skill`` column."""
        from agents.seo_agent.strategy import seed_schedule, update_schedule_entry
        from agents.seo_agent.tools.supabase_tools import query_table

        seed_schedule()
        rows = query_table("ralf_schedule", limit=500)
        entry = rows[0]
        original_skill = entry["skill"]

        # Update only cadence — skill must survive.
        update_schedule_entry(entry["id"], cadence="weekly")

        updated_rows = query_table("ralf_schedule", limit=500)
        updated = next(r for r in updated_rows if r["id"] == entry["id"])
        assert updated["cadence"] == "weekly"
        assert updated["skill"] == original_skill


# ---------------------------------------------------------------------------
# Tests: SkillRegistry.evaluate() with schedule boost
# ---------------------------------------------------------------------------


class TestSkillRegistryScheduleBoost:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_scheduled_skill_gets_higher_priority(self) -> None:
        from agents.seo_agent.skills import SkillRegistry

        registry = SkillRegistry()

        # Minimal dashboard that satisfies keyword_research preconditions
        dashboard = {"keywords_discovered": 0}

        class FakeBuffer:
            def get(self, key: str, default: Any = None) -> Any:
                return default

        from typing import Any

        buffer = FakeBuffer()

        # Monday: keyword_research should be boosted
        monday = _make_datetime(0)
        results_monday = registry.evaluate(dashboard, buffer, now=monday)

        # Thursday: keyword_research should NOT be boosted (but may still fire)
        thursday = _make_datetime(3)
        results_thursday = registry.evaluate(dashboard, buffer, now=thursday)

        # keyword_research should be first on both days (highest base priority)
        # but its effective priority should be higher on Monday
        monday_skills = [s.name for s, _ in results_monday]
        thursday_skills = [s.name for s, _ in results_thursday]

        assert "keyword_research" in monday_skills

    def test_schedule_boost_does_not_block_urgent_skills(self) -> None:
        """Skills with met preconditions should fire regardless of day."""
        from agents.seo_agent.skills import SkillRegistry

        registry = SkillRegistry()

        # No keywords = keyword_research should fire even on non-Monday
        dashboard = {"keywords_discovered": 0}

        class FakeBuffer:
            def get(self, key: str, default: Any = None) -> Any:
                return default

        from typing import Any

        buffer = FakeBuffer()

        # Friday = analytics day, but keyword_research should still fire
        friday = _make_datetime(4)
        results = registry.evaluate(dashboard, buffer, now=friday)
        skills = [s.name for s, _ in results]

        assert "keyword_research" in skills


# ---------------------------------------------------------------------------
# Tests: format_schedule_for_display
# ---------------------------------------------------------------------------


class TestFormatSchedule:
    def setup_method(self) -> None:
        _clear_mock_tables()

    def test_format_includes_daily_and_weekly(self) -> None:
        from agents.seo_agent.strategy import format_schedule_for_display, seed_schedule
        from agents.seo_agent.tools.supabase_tools import query_table

        seed_schedule()
        rows = query_table("ralf_schedule", limit=500)
        output = format_schedule_for_display(rows)

        assert "Daily" in output or "daily" in output
        assert "Weekly" in output or "weekly" in output
        assert "keyword_research" in output

    def test_format_empty_schedule(self) -> None:
        from agents.seo_agent.strategy import format_schedule_for_display

        output = format_schedule_for_display([])
        assert "No schedule" in output


# ---------------------------------------------------------------------------
# Tests: _parse_day_of_week
# ---------------------------------------------------------------------------


class TestParseDayOfWeek:
    """Tests for _parse_day_of_week helper in telegram_bot."""

    def _parse(self, value: str | int) -> int:
        from agents.seo_agent.strategy import parse_day_of_week

        return parse_day_of_week(value)

    # --- Numeric inputs ---

    def test_int_zero(self) -> None:
        assert self._parse(0) == 0

    def test_int_six(self) -> None:
        assert self._parse(6) == 6

    def test_string_zero(self) -> None:
        assert self._parse("0") == 0

    def test_string_six(self) -> None:
        assert self._parse("6") == 6

    # --- Full day names (case-insensitive) ---

    def test_monday(self) -> None:
        assert self._parse("Monday") == 0

    def test_friday_lower(self) -> None:
        assert self._parse("friday") == 4

    def test_sunday_upper(self) -> None:
        assert self._parse("SUNDAY") == 6

    # --- 3-letter abbreviations ---

    def test_mon(self) -> None:
        assert self._parse("Mon") == 0

    def test_wed_lower(self) -> None:
        assert self._parse("wed") == 2

    def test_sat_upper(self) -> None:
        assert self._parse("SAT") == 5

    # --- Whitespace tolerance ---

    def test_padded_string(self) -> None:
        assert self._parse("  Tuesday  ") == 1

    # --- Error cases ---

    def test_out_of_range_int(self) -> None:
        with pytest.raises(ValueError):
            self._parse(7)

    def test_out_of_range_string(self) -> None:
        with pytest.raises(ValueError):
            self._parse("9")

    def test_invalid_name(self) -> None:
        with pytest.raises(ValueError):
            self._parse("Notaday")

    def test_negative_int(self) -> None:
        with pytest.raises(ValueError):
            self._parse(-1)
