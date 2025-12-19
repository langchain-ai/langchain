"""Tests for ACE (Agentic Context Engineering) middleware."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents.middleware.ace import (
    ACEMiddleware,
    ACEPlaybook,
    SectionName,
    extract_bullet_ids,
    format_playbook_line,
    get_playbook_stats,
    initialize_empty_playbook,
    parse_playbook_line,
    update_bullet_counts,
)
from langchain.agents.middleware.ace.playbook import (
    _normalize_section_name,
    add_bullet_to_playbook,
    count_tokens_approximate,
    extract_playbook_bullets,
    get_section_slug,
    limit_playbook_to_budget,
    prune_harmful_bullets,
)
from langchain.agents.middleware.ace.prompts import (
    build_curator_prompt,
    build_reflector_prompt,
    build_system_prompt_with_playbook,
)


class MockReflectorModel(BaseChatModel):
    """Mock model that returns reflector-style JSON responses."""

    def _generate(self, messages: Any, **kwargs: Any) -> ChatResult:
        response = json.dumps(
            {
                "analysis": "Good response",
                "what_worked": "Used correct approach",
                "what_failed": "Nothing significant",
                "key_insight": "Always verify inputs before processing",
                "bullet_tags": [{"id": "str-00001", "tag": "helpful"}],
            }
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "mock-reflector"


class MockCuratorModel(BaseChatModel):
    """Mock model that returns curator-style JSON responses."""

    def _generate(self, messages: Any, **kwargs: Any) -> ChatResult:
        response = json.dumps(
            {
                "reasoning": "Adding new insight from reflection",
                "operations": [
                    {
                        "type": "ADD",
                        "section": SectionName.STRATEGIES_AND_INSIGHTS.value,
                        "content": "New strategy learned from interaction",
                        "reason": "Derived from reflection",
                    }
                ],
            }
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "mock-curator"


class MockEmptyResponseModel(BaseChatModel):
    """Mock model that returns empty/invalid responses."""

    def _generate(self, messages: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Invalid response"))]
        )

    @property
    def _llm_type(self) -> str:
        return "mock-empty"


# =============================================================================
# Playbook Utility Tests
# =============================================================================


class TestPlaybookParsing:
    """Tests for playbook parsing utilities."""

    def test_parse_playbook_line_valid(self) -> None:
        """Test parsing a valid playbook line."""
        line = "[str-00001] helpful=5 harmful=2 :: Always verify data types"
        result = parse_playbook_line(line)

        assert result is not None
        assert result.id == "str-00001"
        assert result.helpful == 5
        assert result.harmful == 2
        assert result.content == "Always verify data types"

    def test_parse_playbook_line_with_whitespace(self) -> None:
        """Test parsing with extra whitespace."""
        line = "  [cal-00042] helpful=10 harmful=0 :: Use formula X = Y + Z  "
        result = parse_playbook_line(line)

        assert result is not None
        assert result.id == "cal-00042"
        assert result.helpful == 10
        assert result.harmful == 0

    def test_parse_playbook_line_invalid(self) -> None:
        """Test parsing invalid lines returns None."""
        assert parse_playbook_line("## SECTION HEADER") is None
        assert parse_playbook_line("") is None
        assert parse_playbook_line("Just some text") is None
        assert parse_playbook_line("[invalid format] content") is None

    def test_format_playbook_line(self) -> None:
        """Test formatting a playbook line."""
        result = format_playbook_line("str-00001", 5, 2, "Test content")
        assert result == "[str-00001] helpful=5 harmful=2 :: Test content"

    def test_extract_bullet_ids(self) -> None:
        """Test extracting bullet IDs from text."""
        text = "I used [str-00001] and [cal-00002] for this answer."
        ids = extract_bullet_ids(text)

        assert len(ids) == 2
        assert "str-00001" in ids
        assert "cal-00002" in ids

    def test_extract_bullet_ids_no_matches(self) -> None:
        """Test extracting when no bullet IDs present."""
        text = "No bullet references here."
        ids = extract_bullet_ids(text)
        assert len(ids) == 0

    def test_get_section_slug(self) -> None:
        """Test section slug mapping."""
        # Test with enum
        assert get_section_slug(SectionName.STRATEGIES_AND_INSIGHTS) == "str"
        assert get_section_slug(SectionName.FORMULAS_AND_CALCULATIONS) == "cal"
        assert get_section_slug(SectionName.COMMON_MISTAKES_TO_AVOID) == "mis"
        # Test with string (backwards compatibility for LLM output)
        assert get_section_slug("strategies_and_insights") == "str"
        assert get_section_slug("STRATEGIES & INSIGHTS") == "str"
        assert get_section_slug("unknown_section") == "oth"


class TestSectionNameNormalization:
    """Tests for section name normalization (regression tests for curator output handling)."""

    def test_normalize_strips_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        assert _normalize_section_name("  STRATEGIES & INSIGHTS  ") == "strategies_and_insights"
        assert _normalize_section_name("STRATEGIES & INSIGHTS\n") == "strategies_and_insights"
        assert _normalize_section_name("\t OTHERS \t") == "others"

    def test_normalize_handles_hyphens(self) -> None:
        """Test that hyphens are treated like spaces/underscores."""
        # "PROBLEM-SOLVING HEURISTICS" from playbook header
        assert _normalize_section_name("PROBLEM-SOLVING HEURISTICS") == "problem_solving_heuristics"
        # "Problem Solving Heuristics" without hyphen (common LLM variation)
        assert _normalize_section_name("Problem Solving Heuristics") == "problem_solving_heuristics"
        # Both should produce the same result
        assert _normalize_section_name("PROBLEM-SOLVING HEURISTICS") == _normalize_section_name(
            "Problem Solving Heuristics"
        )

    def test_normalize_handles_ampersands(self) -> None:
        """Test that ampersands are normalized to 'and'."""
        assert _normalize_section_name("STRATEGIES & INSIGHTS") == "strategies_and_insights"
        assert _normalize_section_name("strategies and insights") == "strategies_and_insights"
        assert _normalize_section_name("CODE SNIPPETS & TEMPLATES") == "code_snippets_and_templates"

    def test_normalize_collapses_multiple_underscores(self) -> None:
        """Test that multiple separators collapse to single underscore."""
        assert _normalize_section_name("code   snippets") == "code_snippets"
        assert _normalize_section_name("code - snippets") == "code_snippets"
        assert _normalize_section_name("code--snippets") == "code_snippets"

    def test_get_section_slug_with_trailing_newline(self) -> None:
        """Regression test: LLM output often has trailing newlines."""
        assert get_section_slug("STRATEGIES & INSIGHTS\n") == "str"
        assert get_section_slug("FORMULAS & CALCULATIONS\r\n") == "cal"
        assert get_section_slug("  OTHERS  \n") == "oth"

    def test_get_section_slug_without_hyphen(self) -> None:
        """Regression test: LLM may omit hyphens from 'PROBLEM-SOLVING HEURISTICS'."""
        # With hyphen (as in playbook)
        assert get_section_slug("PROBLEM-SOLVING HEURISTICS") == "heu"
        # Without hyphen (common LLM variation)
        assert get_section_slug("Problem Solving Heuristics") == "heu"
        assert get_section_slug("problem solving heuristics") == "heu"

    def test_add_bullet_with_trailing_whitespace(self) -> None:
        """Regression test: curator output may include trailing newlines in section."""
        playbook = initialize_empty_playbook()

        # Section name with trailing newline (common LLM output issue)
        updated, next_id = add_bullet_to_playbook(
            playbook, "strategies_and_insights\n", "Test strategy", 1
        )

        # Should match the section and use correct slug
        assert "[str-00001]" in updated
        assert "Test strategy" in updated
        assert next_id == 2

        # Test with enum (preferred API)
        updated2, next_id2 = add_bullet_to_playbook(
            playbook, SectionName.STRATEGIES_AND_INSIGHTS, "Another strategy", 1
        )
        assert "[str-00001]" in updated2
        assert "Another strategy" in updated2
        assert next_id2 == 2

    def test_add_bullet_without_hyphen_in_section(self) -> None:
        """Regression test: curator may output 'Problem Solving Heuristics' without hyphen."""
        playbook = initialize_empty_playbook()

        # Section name without hyphen (LLM variation)
        updated, _ = add_bullet_to_playbook(
            playbook, "Problem Solving Heuristics", "New heuristic", 1
        )

        # Should match "## problem_solving_heuristics" header and use 'heu' slug
        assert "[heu-00001]" in updated
        assert "New heuristic" in updated
        # Verify it was added under the correct section (not OTHERS)
        lines = updated.split("\n")
        heuristics_idx = next(
            i for i, line in enumerate(lines) if "problem_solving_heuristics" in line
        )
        content_idx = next(i for i, line in enumerate(lines) if "New heuristic" in line)
        others_idx = next(i for i, line in enumerate(lines) if line.strip() == "## others")
        # Content should appear after heuristics header but before OTHERS
        assert heuristics_idx < content_idx < others_idx

    def test_add_bullet_mixed_case_and_separators(self) -> None:
        """Test various case and separator combinations all match correctly."""
        playbook = initialize_empty_playbook()

        test_cases = [
            ("STRATEGIES & INSIGHTS", "str"),
            ("strategies & insights", "str"),
            ("Strategies And Insights", "str"),
            ("strategies_and_insights", "str"),
            ("FORMULAS & CALCULATIONS", "cal"),
            ("formulas and calculations", "cal"),
            ("CODE SNIPPETS & TEMPLATES", "cod"),
            ("Code Snippets And Templates", "cod"),
            ("COMMON MISTAKES TO AVOID", "mis"),
            ("common mistakes to avoid", "mis"),
            ("PROBLEM-SOLVING HEURISTICS", "heu"),
            ("problem-solving heuristics", "heu"),
            ("Problem Solving Heuristics", "heu"),  # Without hyphen
            ("CONTEXT CLUES & INDICATORS", "ctx"),
            ("context clues and indicators", "ctx"),
        ]

        for section_name, expected_slug in test_cases:
            updated, _ = add_bullet_to_playbook(playbook, section_name, f"Test {section_name}", 1)
            assert f"[{expected_slug}-00001]" in updated, (
                f"Section '{section_name}' should produce slug '{expected_slug}'"
            )


class TestPlaybookOperations:
    """Tests for playbook modification operations."""

    def test_initialize_empty_playbook(self) -> None:
        """Test creating an empty playbook."""
        playbook = initialize_empty_playbook()

        # Now uses normalized snake_case section names
        assert "## strategies_and_insights" in playbook
        assert "## formulas_and_calculations" in playbook
        assert "## common_mistakes_to_avoid" in playbook
        assert "## others" in playbook

    def test_update_bullet_counts_helpful(self) -> None:
        """Test updating bullet counts with helpful tag."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Test bullet"""

        tags = [{"id": "str-00001", "tag": "helpful"}]
        updated = update_bullet_counts(playbook, tags)

        assert "helpful=1" in updated
        assert "harmful=0" in updated

    def test_update_bullet_counts_harmful(self) -> None:
        """Test updating bullet counts with harmful tag."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Test bullet"""

        tags = [{"id": "str-00001", "tag": "harmful"}]
        updated = update_bullet_counts(playbook, tags)

        assert "helpful=0" in updated
        assert "harmful=1" in updated

    def test_update_bullet_counts_neutral(self) -> None:
        """Test that neutral tag doesn't change counts."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=2 :: Test bullet"""

        tags = [{"id": "str-00001", "tag": "neutral"}]
        updated = update_bullet_counts(playbook, tags)

        assert "helpful=5" in updated
        assert "harmful=2" in updated

    def test_update_bullet_counts_empty_tags(self) -> None:
        """Test with empty tags list."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Test bullet"""

        updated = update_bullet_counts(playbook, [])
        assert updated == playbook

    def test_add_bullet_to_playbook(self) -> None:
        """Test adding a new bullet to the playbook."""
        playbook = initialize_empty_playbook()

        updated, next_id = add_bullet_to_playbook(
            playbook, "STRATEGIES & INSIGHTS", "New strategy content", 1
        )

        assert "[str-00001]" in updated
        assert "New strategy content" in updated
        assert next_id == 2

    def test_add_bullet_to_unknown_section(self) -> None:
        """Test adding bullet to non-existent section falls back to OTHERS."""
        playbook = initialize_empty_playbook()

        updated, next_id = add_bullet_to_playbook(
            playbook, "NONEXISTENT SECTION", "Test content", 1
        )

        # Should be added somewhere (falls back to OTHERS)
        assert "Test content" in updated
        assert next_id == 2

    def test_extract_playbook_bullets(self) -> None:
        """Test extracting specific bullets from playbook."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: First strategy
[str-00002] helpful=3 harmful=1 :: Second strategy

## FORMULAS & CALCULATIONS
[cal-00003] helpful=2 harmful=0 :: Important formula"""

        result = extract_playbook_bullets(playbook, ["str-00001", "cal-00003"])

        assert "[str-00001]" in result
        assert "First strategy" in result
        assert "[cal-00003]" in result
        assert "Important formula" in result
        assert "[str-00002]" not in result

    def test_extract_playbook_bullets_empty_ids(self) -> None:
        """Test extracting with empty ID list."""
        playbook = initialize_empty_playbook()
        result = extract_playbook_bullets(playbook, [])
        assert "No bullets referenced" in result

    def test_prune_harmful_bullets(self) -> None:
        """Test pruning predominantly harmful bullets."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=1 harmful=5 :: Bad advice
[str-00002] helpful=5 harmful=1 :: Good advice"""

        pruned = prune_harmful_bullets(playbook, threshold=0.5, min_interactions=3)

        assert "[str-00001]" not in pruned
        assert "[str-00002]" in pruned

    def test_prune_harmful_bullets_min_interactions(self) -> None:
        """Test that bullets with few interactions are not pruned."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=2 :: New but harmful"""

        # With min_interactions=5, should not be pruned
        pruned = prune_harmful_bullets(playbook, threshold=0.5, min_interactions=5)
        assert "[str-00001]" in pruned

    def test_count_tokens_approximate(self) -> None:
        """Test approximate token counting."""
        # ~4 chars per token
        text = "a" * 100  # Should be ~25 tokens
        tokens = count_tokens_approximate(text)
        assert tokens == 25

        # Empty string
        assert count_tokens_approximate("") == 0

        # Longer text
        text = "a" * 400  # Should be ~100 tokens
        assert count_tokens_approximate(text) == 100

    def test_limit_playbook_to_budget_under_budget(self) -> None:
        """Test that playbook under budget is returned unchanged."""
        playbook = """## strategies_and_insights
[str-00001] helpful=5 harmful=0 :: Short insight"""

        # Budget of 1000 tokens should keep everything
        result = limit_playbook_to_budget(playbook, token_budget=1000)
        assert result == playbook

    def test_limit_playbook_to_budget_prioritizes_helpful(self) -> None:
        """Test that limiting prioritizes high-performing bullets."""
        playbook = """## strategies_and_insights
[str-00001] helpful=10 harmful=0 :: Best bullet
[str-00002] helpful=5 harmful=3 :: Medium bullet
[str-00003] helpful=0 harmful=5 :: Worst bullet"""

        # Very low budget to force trimming
        # Header: ~6 tokens, each bullet: ~12 tokens
        # With budget of 25 and reserve of 5, only ~14 tokens available for bullets
        # Should keep only the best bullet (str-00001)
        result = limit_playbook_to_budget(playbook, token_budget=25, reserve_tokens=5)

        assert "[str-00001]" in result  # Best bullet kept
        assert "[str-00003]" not in result  # Worst bullet dropped

    def test_limit_playbook_to_budget_preserves_sections(self) -> None:
        """Test that section structure is preserved when limiting."""
        playbook = """## strategies_and_insights
[str-00001] helpful=5 harmful=0 :: Strategy
## common_mistakes_to_avoid
[mis-00001] helpful=3 harmful=0 :: Mistake"""

        # Enough budget to keep both
        result = limit_playbook_to_budget(playbook, token_budget=500)

        assert "## strategies_and_insights" in result
        assert "## common_mistakes_to_avoid" in result

    def test_limit_playbook_to_budget_protects_fresh_bullets(self) -> None:
        """Test that fresh bullets (0 interactions) are prioritized over proven ones.

        This prevents the cold-start problem where newly curated bullets would
        be immediately dropped because they have no votes yet. Fresh bullets
        must survive at least one round to accumulate helpful/harmful counts.
        """
        playbook = """## strategies_and_insights
[str-00001] helpful=10 harmful=0 :: Proven best bullet
[str-00002] helpful=5 harmful=1 :: Proven good bullet
[str-00003] helpful=0 harmful=0 :: Fresh bullet from curator"""

        # Budget that can only fit ~2 bullets
        # Fresh bullet (str-00003) should be kept even though it has lowest "score"
        result = limit_playbook_to_budget(playbook, token_budget=40, reserve_tokens=5)

        # Fresh bullet MUST be included (cold-start protection)
        assert "[str-00003]" in result, "Fresh bullets should survive limiting"
        # Best proven bullet should also be included
        assert "[str-00001]" in result, "Best proven bullet should be kept"


class TestPlaybookStats:
    """Tests for playbook statistics."""

    def test_get_playbook_stats_empty(self) -> None:
        """Test stats for empty playbook."""
        playbook = initialize_empty_playbook()
        stats = get_playbook_stats(playbook)

        assert stats["total_bullets"] == 0
        assert stats["high_performing"] == 0
        assert stats["problematic"] == 0
        assert stats["unused"] == 0

    def test_get_playbook_stats_with_bullets(self) -> None:
        """Test stats with various bullet types."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=10 harmful=0 :: High performer
[str-00002] helpful=0 harmful=5 :: Problematic
[str-00003] helpful=0 harmful=0 :: Unused"""

        stats = get_playbook_stats(playbook)

        assert stats["total_bullets"] == 3
        assert stats["high_performing"] == 1
        assert stats["problematic"] == 1
        assert stats["unused"] == 1

    def test_get_playbook_stats_by_section(self) -> None:
        """Test stats are broken down by section."""
        playbook = """## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Strategy

## FORMULAS & CALCULATIONS
[cal-00001] helpful=3 harmful=1 :: Formula"""

        stats = get_playbook_stats(playbook)

        assert "STRATEGIES & INSIGHTS" in stats["by_section"]
        assert "FORMULAS & CALCULATIONS" in stats["by_section"]
        assert stats["by_section"]["STRATEGIES & INSIGHTS"]["count"] == 1


class TestACEPlaybookDataclass:
    """Tests for ACEPlaybook dataclass."""

    def test_default_values(self) -> None:
        """Test default initialization."""
        playbook = ACEPlaybook()

        # Now uses normalized snake_case section names
        assert "## strategies_and_insights" in playbook.content
        assert playbook.next_global_id == 1
        assert playbook.stats == {}

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        playbook = ACEPlaybook(
            content="test content",
            next_global_id=5,
            stats={"total_bullets": 3},
        )

        d = playbook.to_dict()

        assert d["content"] == "test content"
        assert d["next_global_id"] == 5
        assert d["stats"]["total_bullets"] == 3

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "content": "test content",
            "next_global_id": 10,
            "stats": {"total": 5},
        }

        playbook = ACEPlaybook.from_dict(data)

        assert playbook.content == "test content"
        assert playbook.next_global_id == 10
        assert playbook.stats["total"] == 5


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPrompts:
    """Tests for ACE prompt generation."""

    def test_build_system_prompt_with_playbook_string(self) -> None:
        """Test building prompt from string."""
        result = build_system_prompt_with_playbook(
            original_prompt="You are helpful.",
            playbook="[str-00001] helpful=1 harmful=0 :: Test",
            reflection="",
        )

        assert "You are helpful." in result
        assert "ACE PLAYBOOK" in result
        assert "[str-00001]" in result

    def test_build_system_prompt_with_playbook_none(self) -> None:
        """Test building prompt with None original."""
        result = build_system_prompt_with_playbook(
            original_prompt=None,
            playbook="Test playbook",
            reflection="",
        )

        assert "helpful AI assistant" in result
        assert "Test playbook" in result

    def test_build_system_prompt_with_reflection(self) -> None:
        """Test including reflection in prompt."""
        result = build_system_prompt_with_playbook(
            original_prompt="Base prompt",
            playbook="Playbook content",
            reflection="Previous attempt failed because of X",
        )

        assert "PREVIOUS REFLECTION" in result
        assert "Previous attempt failed because of X" in result

    def test_build_system_prompt_empty_reflection(self) -> None:
        """Test that empty reflection is not included."""
        result = build_system_prompt_with_playbook(
            original_prompt="Base prompt",
            playbook="Playbook content",
            reflection="(empty)",
        )

        assert "PREVIOUS REFLECTION" not in result

    def test_build_reflector_prompt(self) -> None:
        """Test building reflector prompt."""
        result = build_reflector_prompt(
            question="What is 2+2?",
            reasoning_trace="Let me calculate...",
            feedback="Response was correct",
            bullets_used="[str-00001] :: Test bullet",
        )

        assert "What is 2+2?" in result
        assert "Let me calculate" in result
        assert "Response was correct" in result
        assert "[str-00001]" in result

    def test_build_reflector_prompt_with_ground_truth(self) -> None:
        """Test building reflector prompt with ground truth included."""
        result = build_reflector_prompt(
            question="What is 2+2?",
            reasoning_trace="Let me calculate... The answer is 4.",
            feedback="Response was correct",
            bullets_used="[str-00001] :: Test bullet",
            ground_truth="4",
        )

        # Check all expected sections are present
        assert "What is 2+2?" in result
        assert "Let me calculate" in result
        assert "Response was correct" in result
        assert "[str-00001]" in result
        # Ground truth section should be included
        assert "Ground Truth Answer" in result
        assert "4" in result

    def test_build_reflector_prompt_without_ground_truth(self) -> None:
        """Test that ground truth section is absent when not provided."""
        result = build_reflector_prompt(
            question="What is 2+2?",
            reasoning_trace="Let me calculate...",
            feedback="Response was correct",
            bullets_used="[str-00001] :: Test bullet",
            ground_truth=None,
        )

        # Ground truth section should NOT be present
        assert "Ground Truth Answer" not in result
        # Other sections should still be present
        assert "What is 2+2?" in result
        assert "Feedback/Outcome" in result

    def test_build_reflector_prompt_ground_truth_empty_string(self) -> None:
        """Test that empty string ground truth is treated as no ground truth."""
        result = build_reflector_prompt(
            question="What is 2+2?",
            reasoning_trace="Let me calculate...",
            feedback="Response was correct",
            bullets_used="[str-00001] :: Test bullet",
            ground_truth="",
        )

        # Empty string is falsy, so ground truth section should NOT be present
        assert "Ground Truth Answer" not in result

    def test_build_curator_prompt(self) -> None:
        """Test building curator prompt."""
        result = build_curator_prompt(
            current_step=5,
            total_samples=100,
            token_budget=50000,
            playbook_stats='{"total_bullets": 10}',
            recent_reflection="Learned something new",
            current_playbook="## STRATEGIES\n[str-00001]...",
        )

        assert "Step 5 of 100" in result
        assert "50000" in result
        assert "Learned something new" in result


# =============================================================================
# Middleware Tests
# =============================================================================


class TestACEMiddlewareInitialization:
    """Tests for ACE middleware initialization."""

    def test_default_initialization(self) -> None:
        """Test middleware with required models."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        assert middleware.curator_frequency == 5
        assert middleware.playbook_token_budget == 80000
        assert middleware.auto_prune is False

    def test_initialization_with_string_models(self) -> None:
        """Test middleware with model name strings."""
        with patch(
            "langchain.agents.middleware.ace.middleware.init_chat_model",
            return_value=MockReflectorModel(),
        ):
            middleware = ACEMiddleware(
                reflector_model="gpt-4o-mini",
                curator_model="gpt-4o-mini",
                curator_frequency=10,
            )

            assert middleware.curator_frequency == 10

    def test_initialization_with_initial_playbook(self) -> None:
        """Test middleware with custom initial playbook."""
        custom_playbook = """## strategies_and_insights
[str-00001] helpful=5 harmful=0 :: Custom strategy"""

        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook=custom_playbook,
        )

        assert "[str-00001]" in middleware.initial_playbook
        assert "Custom strategy" in middleware.initial_playbook

    def test_initialization_with_auto_prune(self) -> None:
        """Test middleware with auto-pruning enabled."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            auto_prune=True,
            prune_threshold=0.6,
            prune_min_interactions=5,
        )

        assert middleware.auto_prune is True
        assert middleware.prune_threshold == 0.6
        assert middleware.prune_min_interactions == 5


class TestACEMiddlewareHooks:
    """Tests for ACE middleware hook methods."""

    def test_before_agent_initializes_state(self) -> None:
        """Test that before_agent initializes ACE state fields."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        state: dict[str, Any] = {"messages": []}
        result = middleware.before_agent(state, None)  # type: ignore[arg-type]

        assert result is not None
        assert "ace_playbook" in result
        assert "ace_last_reflection" in result
        assert "ace_interaction_count" in result
        assert result["ace_interaction_count"] == 0

    def test_before_agent_does_not_reinitialize(self) -> None:
        """Test that before_agent doesn't overwrite existing state."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        state: dict[str, Any] = {
            "messages": [],
            "ace_playbook": {"content": "existing", "next_global_id": 5, "stats": {}},
        }
        result = middleware.before_agent(state, None)  # type: ignore[arg-type]

        assert result is None

    def test_wrap_model_call_injects_playbook(self) -> None:
        """Test that wrap_model_call injects playbook into system prompt."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook="[str-00001] helpful=1 harmful=0 :: Test strategy",
        )

        from langchain_core.messages import SystemMessage

        from langchain.agents.middleware.types import ModelRequest, ModelResponse

        # Create mock request
        class MockModel:
            pass

        request = ModelRequest(
            model=MockModel(),  # type: ignore[arg-type]
            messages=[HumanMessage(content="Hello")],
            system_message=SystemMessage(content="Be helpful"),
            state={
                "messages": [],
                "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
                "ace_last_reflection": "",
            },
        )

        # Track what gets passed to handler
        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="Response")])

        middleware.wrap_model_call(request, mock_handler)

        assert captured_request is not None
        assert captured_request.system_message is not None
        system_content = captured_request.system_message.content
        assert "ACE PLAYBOOK" in system_content
        assert "[str-00001]" in system_content

    def test_after_model_with_reflection(self) -> None:
        """Test after_model runs reflector and updates state."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook="""## strategies_and_insights
[str-00001] helpful=0 harmful=0 :: Test strategy""",
        )

        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="What is the answer?"),
                AIMessage(content="The answer uses [str-00001] strategy."),
            ],
            "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
            "ace_last_reflection": "",
            "ace_interaction_count": 0,
        }

        result = middleware.after_model(state, None)  # type: ignore[arg-type]

        assert result is not None
        assert "ace_playbook" in result
        assert "ace_last_reflection" in result
        assert result["ace_interaction_count"] == 1

        # Check that reflection was captured
        assert "verify inputs" in result["ace_last_reflection"]

    def test_after_model_handles_non_ai_message(self) -> None:
        """Test after_model handles case where last message is not AI."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        state: dict[str, Any] = {
            "messages": [HumanMessage(content="User message")],
            "ace_interaction_count": 0,
        }

        result = middleware.after_model(state, None)  # type: ignore[arg-type]

        # Should return None (no state updates) when there's no AI response to analyze
        assert result is None

    def test_after_model_skips_counter_for_pending_tool_calls(self) -> None:
        """Test that interaction counter is NOT incremented when AI message has pending tool_calls.

        This is critical: ace_interaction_count should measure "completed agent interactions",
        not "raw LLM invocations". When the agent is in a tool loop (has pending tool_calls),
        the counter should not advance, ensuring curator_frequency, expected_interactions,
        and pruning thresholds work correctly.
        """
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        # AI message with pending tool calls (non-terminal response)
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="What's the weather?"),
                AIMessage(
                    content="Let me check the weather for you.",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    ],
                ),
            ],
            "ace_interaction_count": 5,
        }

        result = middleware.after_model(state, None)  # type: ignore[arg-type]

        # Should return None (no state updates) - counter should NOT increment
        # This prevents counter drift during tool loops
        assert result is None

    def test_ground_truth_passed_to_reflector(self) -> None:
        """Test that ground_truth is passed from state to reflector prompt."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook=initialize_empty_playbook(),
        )

        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="The answer is 4."),
            ],
            "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
            "ace_last_reflection": "",
            "ace_interaction_count": 0,
            "ground_truth": "4",  # Ground truth provided for evaluation
        }

        # Run _prepare_reflection_context to get the reflector prompt
        context = middleware._prepare_reflection_context(state)  # type: ignore[arg-type]

        assert context is not None
        _, reflector_prompt, _, _ = context

        # Ground truth section should be in the prompt
        assert "Ground Truth Answer" in reflector_prompt
        assert "4" in reflector_prompt

    def test_ground_truth_none_excludes_section(self) -> None:
        """Test that missing ground_truth excludes GT section from prompt."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook=initialize_empty_playbook(),
        )

        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="The answer is 4."),
            ],
            "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
            "ace_last_reflection": "",
            "ace_interaction_count": 0,
            # No ground_truth field
        }

        context = middleware._prepare_reflection_context(state)  # type: ignore[arg-type]

        assert context is not None
        _, reflector_prompt, _, _ = context

        # Ground truth section should NOT be in the prompt
        assert "Ground Truth Answer" not in reflector_prompt


class TestACEMiddlewareCuration:
    """Tests for ACE middleware curation functionality."""

    def test_curator_runs_at_frequency(self) -> None:
        """Test that curator runs at specified frequency."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            curator_frequency=2,
            initial_playbook=initialize_empty_playbook(),
        )

        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Answer"),
            ],
            "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
            "ace_last_reflection": "",
            "ace_interaction_count": 1,  # Next will be 2, triggering curator
        }

        result = middleware.after_model(state, None)  # type: ignore[arg-type]

        assert result is not None
        # Check that curator added a new bullet
        playbook_content = result["ace_playbook"]["content"]
        assert "New strategy learned" in playbook_content

    def test_curator_does_not_run_before_frequency(self) -> None:
        """Test that curator doesn't run before frequency threshold."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            curator_frequency=5,
            initial_playbook=initialize_empty_playbook(),
        )

        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Answer"),
            ],
            "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
            "ace_last_reflection": "",
            "ace_interaction_count": 2,  # Next will be 3, not triggering
        }

        result = middleware.after_model(state, None)  # type: ignore[arg-type]

        # Curator should not have run
        playbook_content = result["ace_playbook"]["content"]
        assert "New strategy learned" not in playbook_content


class TestACEMiddlewareJSONParsing:
    """Tests for JSON parsing in middleware."""

    def test_extract_json_direct_parse(self) -> None:
        """Test extracting JSON from direct JSON string."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        json_str = '{"key": "value", "nested": {"a": 1}}'
        result = middleware._extract_json_from_response(json_str)

        assert result is not None
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_extract_json_from_code_block(self) -> None:
        """Test extracting JSON from code block."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        text = """Here is the response:
```json
{"analysis": "good", "score": 10}
```
End of response."""

        result = middleware._extract_json_from_response(text)

        assert result is not None
        assert result["analysis"] == "good"
        assert result["score"] == 10

    def test_extract_json_from_embedded(self) -> None:
        """Test extracting JSON embedded in text."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        text = 'Some text before {"data": "value"} some text after'
        result = middleware._extract_json_from_response(text)

        assert result is not None
        assert result["data"] == "value"

    def test_extract_json_invalid_returns_none(self) -> None:
        """Test that invalid JSON returns None."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        result = middleware._extract_json_from_response("Not JSON at all")
        assert result is None


class TestACEMiddlewareAsync:
    """Tests for async middleware methods."""

    @pytest.mark.asyncio
    async def test_abefore_agent(self) -> None:
        """Test async before_agent."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
        )

        state: dict[str, Any] = {"messages": []}
        result = await middleware.abefore_agent(state, None)  # type: ignore[arg-type]

        assert result is not None
        assert "ace_playbook" in result

    @pytest.mark.asyncio
    async def test_awrap_model_call(self) -> None:
        """Test async wrap_model_call."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook="[str-00001] helpful=1 harmful=0 :: Strategy",
        )

        from langchain_core.messages import SystemMessage

        from langchain.agents.middleware.types import ModelRequest, ModelResponse

        class MockModel:
            pass

        request = ModelRequest(
            model=MockModel(),  # type: ignore[arg-type]
            messages=[HumanMessage(content="Hello")],
            system_message=SystemMessage(content="Be helpful"),
            state={
                "messages": [],
                "ace_playbook": middleware._get_playbook({}).to_dict(),  # type: ignore[arg-type]
                "ace_last_reflection": "",
            },
        )

        async def mock_handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[AIMessage(content="Response")])

        result = await middleware.awrap_model_call(request, mock_handler)

        assert result is not None


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestACEMiddlewareIntegration:
    """Integration-style tests for ACE middleware."""

    def test_full_workflow(self) -> None:
        """Test complete workflow with reflection and curation."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            initial_playbook="""## strategies_and_insights
[str-00001] helpful=5 harmful=0 :: Use systematic approach""",
        )

        # Initialize state
        state: dict[str, Any] = {"messages": []}
        init_result = middleware.before_agent(state, None)  # type: ignore[arg-type]
        state.update(init_result or {})

        # Verify initialization
        assert state["ace_interaction_count"] == 0
        assert "[str-00001]" in state["ace_playbook"]["content"]

        # Simulate after model call
        state["messages"] = [
            HumanMessage(content="Question"),
            AIMessage(content="Answer using [str-00001]"),
        ]
        after_result = middleware.after_model(state, None)  # type: ignore[arg-type]
        state.update(after_result or {})

        # Verify interaction count incremented
        assert state["ace_interaction_count"] == 1
        # Verify reflection was captured
        assert state["ace_last_reflection"] != ""

    def test_playbook_evolution_over_interactions(self) -> None:
        """Test that playbook evolves across multiple interactions."""
        middleware = ACEMiddleware(
            reflector_model=MockReflectorModel(),
            curator_model=MockCuratorModel(),
            curator_frequency=1,  # Curate every interaction
            initial_playbook="""## strategies_and_insights
[str-00001] helpful=0 harmful=0 :: Initial strategy""",
        )

        state: dict[str, Any] = {"messages": []}
        init_result = middleware.before_agent(state, None)  # type: ignore[arg-type]
        state.update(init_result or {})

        # First interaction
        state["messages"] = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1 using [str-00001]"),
        ]
        result1 = middleware.after_model(state, None)  # type: ignore[arg-type]
        state.update(result1 or {})

        # Check that curator added new content
        assert "New strategy learned" in state["ace_playbook"]["content"]

        # Check that bullet count was updated
        assert "helpful=1" in state["ace_playbook"]["content"]
