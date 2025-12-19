"""Playbook management utilities for ACE middleware.

The playbook is an evolving context that accumulates strategies, insights,
and patterns learned from agent interactions. Each entry (bullet) tracks
helpful and harmful counts to enable self-improvement.

Playbook format:
    ## SECTION NAME
    [section-00001] helpful=5 harmful=0 :: Content of the bullet point
    [section-00002] helpful=3 harmful=1 :: Another bullet point
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backwards-compatible StrEnum for Python < 3.11."""

        def __str__(self) -> str:
            """Return string representation of the enum value."""
            return str(self.value)


class SectionName(StrEnum):
    """Canonical ACE playbook section names.

    These normalized snake_case names are used throughout the ACE system
    to ensure consistent matching between curator output and playbook headers.
    """

    STRATEGIES_AND_INSIGHTS = "strategies_and_insights"
    FORMULAS_AND_CALCULATIONS = "formulas_and_calculations"
    CODE_SNIPPETS_AND_TEMPLATES = "code_snippets_and_templates"
    COMMON_MISTAKES_TO_AVOID = "common_mistakes_to_avoid"
    PROBLEM_SOLVING_HEURISTICS = "problem_solving_heuristics"
    CONTEXT_CLUES_AND_INDICATORS = "context_clues_and_indicators"
    OTHERS = "others"


# Thresholds for classifying bullet performance
_HIGH_PERFORMING_HELPFUL_MIN = 5  # Minimum helpful count for high-performing bullets
_HIGH_PERFORMING_HARMFUL_MAX = 2  # Maximum harmful count for high-performing bullets

# Section slug mappings for bullet ID generation
_SECTION_SLUGS: dict[str, str] = {
    SectionName.STRATEGIES_AND_INSIGHTS: "str",
    SectionName.FORMULAS_AND_CALCULATIONS: "cal",
    SectionName.CODE_SNIPPETS_AND_TEMPLATES: "cod",
    SectionName.COMMON_MISTAKES_TO_AVOID: "mis",
    SectionName.PROBLEM_SOLVING_HEURISTICS: "heu",
    SectionName.CONTEXT_CLUES_AND_INDICATORS: "ctx",
    SectionName.OTHERS: "oth",
    "general": "gen",  # Legacy fallback
}


def _normalize_section_name(section: str) -> str:
    r"""Normalize a section name for consistent matching.

    Since we now use canonical snake_case section names everywhere,
    this function is primarily for backwards compatibility with:
    - Legacy playbooks using human-readable headers
    - LLM outputs that may use variations

    Args:
        section: Raw section name from curator output or playbook header.

    Returns:
        Normalized section name for matching.

    Examples:
        >>> _normalize_section_name("strategies_and_insights")
        'strategies_and_insights'
        >>> _normalize_section_name("STRATEGIES & INSIGHTS\n")
        'strategies_and_insights'
        >>> _normalize_section_name("Problem Solving Heuristics")
        'problem_solving_heuristics'
    """
    # Strip leading/trailing whitespace first
    normalized = section.strip()
    # Lowercase
    normalized = normalized.lower()
    # Replace ampersands with "and"
    normalized = normalized.replace("&", "and")
    # Treat hyphens and spaces as equivalent separators -> underscore
    normalized = normalized.replace("-", "_").replace(" ", "_")
    # Collapse multiple underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _build_default_playbook() -> str:
    """Build the default empty playbook with canonical section headers."""
    return "\n\n".join(f"## {section.value}" for section in SectionName)


_DEFAULT_PLAYBOOK = _build_default_playbook()


@dataclass
class ACEPlaybook:
    """Represents an evolving ACE playbook with tracking metadata.

    Attributes:
        content: The raw playbook text content.
        next_global_id: Next available ID for new bullets.
        stats: Statistics about the playbook (bullet counts, performance).
    """

    content: str = field(default_factory=lambda: _DEFAULT_PLAYBOOK)
    next_global_id: int = 1
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "content": self.content,
            "next_global_id": self.next_global_id,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACEPlaybook:
        """Create from dictionary."""
        return cls(
            content=data.get("content", _DEFAULT_PLAYBOOK),
            next_global_id=data.get("next_global_id", 1),
            stats=data.get("stats", {}),
        )


@dataclass
class ParsedBullet:
    """A parsed bullet point from the playbook.

    Attributes:
        id: Unique identifier (e.g., "str-00001").
        helpful: Count of times this bullet was helpful.
        harmful: Count of times this bullet was harmful.
        content: The actual content/advice of the bullet.
    """

    id: str
    helpful: int
    harmful: int
    content: str


def initialize_empty_playbook() -> str:
    """Initialize an empty playbook with standard sections.

    Returns:
        Empty playbook template with section headers.
    """
    return _DEFAULT_PLAYBOOK


def get_section_slug(section_name: SectionName | str) -> str:
    """Get the 3-letter slug for a section name.

    Args:
        section_name: Section name enum or string (e.g., SectionName.STRATEGIES_AND_INSIGHTS
            or "strategies_and_insights").

    Returns:
        3-letter slug (e.g., "str").
    """
    if isinstance(section_name, SectionName):
        return _SECTION_SLUGS.get(section_name, "oth")
    normalized = _normalize_section_name(section_name)
    return _SECTION_SLUGS.get(normalized, "oth")


def parse_playbook_line(line: str) -> ParsedBullet | None:
    """Parse a single playbook line to extract components.

    Args:
        line: A line from the playbook.

    Returns:
        Parsed bullet if valid, None otherwise.

    Example:
        >>> parse_playbook_line("[str-00001] helpful=5 harmful=0 :: Always verify data")
        ParsedBullet(id='str-00001', helpful=5, harmful=0, content='Always verify data')
    """
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())

    if match:
        return ParsedBullet(
            id=match.group(1),
            helpful=int(match.group(2)),
            harmful=int(match.group(3)),
            content=match.group(4),
        )
    return None


def _sanitize_bullet_content(content: str) -> str:
    r"""Sanitize bullet content to ensure it stays on a single line.

    Curator output may contain newlines ("First insight.\nSecond sentence.")
    which would break the playbook format since parsing assumes one bullet per line.

    Args:
        content: Raw bullet content that may contain newlines.

    Returns:
        Content with newlines replaced by spaces and whitespace collapsed.
    """
    # Replace newlines and tabs with spaces
    sanitized = content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Collapse multiple spaces to single space
    sanitized = " ".join(sanitized.split())
    return sanitized.strip()


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into playbook line format.

    Args:
        bullet_id: Unique identifier for the bullet.
        helpful: Count of helpful occurrences.
        harmful: Count of harmful occurrences.
        content: The bullet content.

    Returns:
        Formatted playbook line (always single line).
    """
    sanitized = _sanitize_bullet_content(content)
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {sanitized}"


def extract_bullet_ids(text: str) -> list[str]:
    """Extract bullet IDs referenced in text.

    Args:
        text: Text that may contain bullet ID references.

    Returns:
        List of bullet IDs found.
    """
    pattern = r"\[([a-z]{3}-\d{5})\]"
    return re.findall(pattern, text)


def extract_bullet_ids_from_comment(text: str) -> list[str]:
    """Extract bullet IDs from HTML comment format.

    The agent is instructed to include bullet IDs in a comment like:
    <!-- bullet_ids: ["str-00001", "mis-00002"] -->

    Args:
        text: Text that may contain a bullet_ids comment.

    Returns:
        List of bullet IDs found in the comment, or empty list if none.
    """
    pattern = r"<!--\s*bullet_ids:\s*\[(.*?)\]\s*-->"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Extract the content inside the brackets
        ids_str = match.group(1)
        # Extract individual IDs (handles both quoted and unquoted)
        id_pattern = r'"?([a-z]{3}-\d{5})"?'
        return re.findall(id_pattern, ids_str)
    return []


def get_max_bullet_id(playbook_text: str) -> int:
    """Find the maximum numeric ID used in the playbook.

    Args:
        playbook_text: The playbook content.

    Returns:
        The highest ID number found, or 0 if none found.
    """
    pattern = r"\[[a-z]{3}-(\d{5})\]"
    matches = re.findall(pattern, playbook_text)
    if not matches:
        return 0
    return max(int(m) for m in matches)


def extract_playbook_bullets(playbook_text: str, bullet_ids: list[str]) -> str:
    """Extract specific bullet points from playbook based on IDs.

    Args:
        playbook_text: The full playbook text.
        bullet_ids: List of bullet IDs to extract.

    Returns:
        Formatted string containing only the specified bullets.
    """
    if not bullet_ids:
        return "(No bullets referenced)"

    lines = playbook_text.strip().split("\n")
    found_bullets: list[str] = []

    for line in lines:
        if line.strip():
            parsed = parse_playbook_line(line)
            if parsed and parsed.id in bullet_ids:
                found_bullets.append(
                    format_playbook_line(parsed.id, parsed.helpful, parsed.harmful, parsed.content)
                )

    if not found_bullets:
        return "(Referenced bullets not found in playbook)"

    return "\n".join(found_bullets)


def update_bullet_counts(
    playbook_text: str,
    bullet_tags: list[dict[str, str]],
) -> str:
    """Update helpful/harmful counts based on reflector tags.

    Args:
        playbook_text: Current playbook content.
        bullet_tags: List of dicts with 'id' and 'tag' keys.
            Tag can be 'helpful', 'harmful', or 'neutral'.

    Returns:
        Updated playbook text with modified counts.
    """
    if not bullet_tags:
        return playbook_text

    # Create tag lookup
    tag_map: dict[str, str] = {}
    for tag_entry in bullet_tags:
        if isinstance(tag_entry, dict):
            bullet_id = tag_entry.get("id", "")
            tag_value = tag_entry.get("tag", "neutral")
            if bullet_id:
                tag_map[bullet_id] = tag_value

    if not tag_map:
        return playbook_text

    lines = playbook_text.strip().split("\n")
    updated_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if parsed and parsed.id in tag_map:
            tag = tag_map[parsed.id]
            if tag == "helpful":
                parsed.helpful += 1
            elif tag == "harmful":
                parsed.harmful += 1
            # neutral: no change

            new_line = format_playbook_line(
                parsed.id, parsed.helpful, parsed.harmful, parsed.content
            )
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)


def add_bullet_to_playbook(
    playbook_text: str,
    section: SectionName | str,
    content: str,
    next_global_id: int,
) -> tuple[str, int]:
    """Add a new bullet to the playbook.

    Args:
        playbook_text: Current playbook content.
        section: Section name enum or string (e.g., SectionName.STRATEGIES_AND_INSIGHTS
            or "strategies_and_insights").
        content: Content of the new bullet.
        next_global_id: Next available global ID.

    Returns:
        Tuple of (updated_playbook, new_next_global_id).
    """
    slug = get_section_slug(section)
    new_id = f"{slug}-{next_global_id:05d}"
    new_line = format_playbook_line(new_id, 0, 0, content)

    lines = playbook_text.strip().split("\n")
    section_normalized = _normalize_section_name(
        section if isinstance(section, str) else section.value
    )
    added = False

    # Simple approach: find section and append after header
    result_lines: list[str] = []

    for line in lines:
        result_lines.append(line)

        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            normalized = _normalize_section_name(header)
            if normalized == section_normalized:
                # Add the new bullet right after the section header
                result_lines.append(new_line)
                added = True

    if not added:
        # Section not found, add to others
        for i, line in enumerate(result_lines):
            if line.strip() == f"## {SectionName.OTHERS}":
                result_lines.insert(i + 1, new_line)
                added = True
                break

    if not added:
        # No others section, append at end
        result_lines.append(new_line)

    return "\n".join(result_lines), next_global_id + 1


def get_playbook_stats(playbook_text: str) -> dict[str, Any]:
    """Generate statistics about the playbook.

    Args:
        playbook_text: The playbook content.

    Returns:
        Dictionary with statistics including:
        - total_bullets: Total number of bullets
        - high_performing: Bullets exceeding helpful threshold with low harmful count
        - problematic: Bullets where harmful >= helpful
        - unused: Bullets with no helpful or harmful counts
        - by_section: Breakdown by section
    """
    lines = playbook_text.strip().split("\n")
    stats: dict[str, Any] = {
        "total_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }

    current_section = "general"

    for line in lines:
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue

        parsed = parse_playbook_line(line)
        if parsed:
            stats["total_bullets"] += 1

            if (
                parsed.helpful > _HIGH_PERFORMING_HELPFUL_MIN
                and parsed.harmful < _HIGH_PERFORMING_HARMFUL_MAX
            ):
                stats["high_performing"] += 1
            elif parsed.harmful >= parsed.helpful and parsed.harmful > 0:
                stats["problematic"] += 1
            elif parsed.helpful + parsed.harmful == 0:
                stats["unused"] += 1

            if current_section not in stats["by_section"]:
                stats["by_section"][current_section] = {
                    "count": 0,
                    "helpful": 0,
                    "harmful": 0,
                }

            stats["by_section"][current_section]["count"] += 1
            stats["by_section"][current_section]["helpful"] += parsed.helpful
            stats["by_section"][current_section]["harmful"] += parsed.harmful

    return stats


def prune_harmful_bullets(
    playbook_text: str,
    threshold: float = 0.5,
    min_interactions: int = 3,
) -> str:
    """Remove bullets that are predominantly harmful.

    Args:
        playbook_text: Current playbook content.
        threshold: Harmful ratio threshold (harmful/total) above which to prune.
        min_interactions: Minimum interactions before considering pruning.

    Returns:
        Playbook with harmful bullets removed.
    """
    lines = playbook_text.strip().split("\n")
    filtered_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            filtered_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if parsed:
            total = parsed.helpful + parsed.harmful
            if total >= min_interactions:
                harmful_ratio = parsed.harmful / total
                if harmful_ratio > threshold:
                    # Skip this bullet (prune it)
                    continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


# Token counting constants
_CHARS_PER_TOKEN = 4.0  # Approximate: ~4 chars per token for English text


def count_tokens_approximate(text: str) -> int:
    """Approximate token count for text.

    Uses the standard approximation of ~4 characters per token for English text.
    This matches the approach used in the original ACE implementation.

    Args:
        text: Text to count tokens for.

    Returns:
        Approximate number of tokens.
    """
    return int(len(text) / _CHARS_PER_TOKEN)


def _compute_bullet_priority(bullet: ParsedBullet) -> tuple[float, int]:
    """Compute priority score for a bullet for sorting.

    Bullets are prioritized by:
    1. Net score (helpful - harmful), higher is better
    2. Total interactions as tiebreaker (more tested = more reliable)

    Args:
        bullet: Parsed bullet to score.

    Returns:
        Tuple of (net_score, total_interactions) for sorting.
        Higher values = higher priority.
    """
    net_score = bullet.helpful - bullet.harmful
    total_interactions = bullet.helpful + bullet.harmful
    return (net_score, total_interactions)


# Default threshold for considering a bullet "fresh" (needs protection from budget limiting)
BULLET_COUNT_FRESHNESS_THRESHOLD = 1


def _is_fresh_bullet(
    bullet: ParsedBullet,
    threshold: int = BULLET_COUNT_FRESHNESS_THRESHOLD,
) -> bool:
    """Check if a bullet is fresh (few interactions).

    Fresh bullets are newly curated additions that haven't accumulated
    enough feedback yet. They need special handling to ensure they survive
    budget limiting until they've been evaluated sufficiently.

    Args:
        bullet: Parsed bullet to check.
        threshold: Maximum total interactions to be considered fresh.
            Bullets with total interactions < threshold are fresh.
            Default is BULLET_COUNT_FRESHNESS_THRESHOLD (1), meaning
            only bullets with 0 interactions are fresh.

    Returns:
        True if bullet has fewer than threshold total interactions.
    """
    return bullet.helpful + bullet.harmful < threshold


def limit_playbook_to_budget(
    playbook_text: str,
    token_budget: int,
    *,
    reserve_tokens: int = 1000,
) -> str:
    """Limit playbook content to fit within a token budget.

    When the playbook exceeds the token budget, this function retains
    the highest-priority bullets (based on helpful-harmful score) while
    preserving the section structure. This prevents context window
    overflow and prompt bloat.

    **Fresh bullet protection:** Newly curated bullets (with 0 interactions)
    are always included first to ensure they survive at least one round.
    This prevents the "cold start" problem where new insights would be
    immediately dropped because they have no votes yet. See original ACE:
    the reference implementation doesn't programmatically enforce budgets,
    relying on the curator prompt instead.

    Args:
        playbook_text: The full playbook content.
        token_budget: Maximum tokens allowed for the playbook.
        reserve_tokens: Tokens to reserve for section headers and overhead.
            Default 1000 provides buffer for headers and formatting.

    Returns:
        Playbook content trimmed to fit within the budget.
        If already under budget, returns the original content unchanged.
    """
    current_tokens = count_tokens_approximate(playbook_text)
    if current_tokens <= token_budget:
        return playbook_text

    # Parse playbook into sections and bullets
    lines = playbook_text.strip().split("\n")

    # Track sections and their bullets
    sections: list[tuple[str, list[tuple[ParsedBullet, str]]]] = []
    current_section_header: str | None = None
    current_section_bullets: list[tuple[ParsedBullet, str]] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            # Save previous section if exists
            if current_section_header is not None:
                sections.append((current_section_header, current_section_bullets))
            current_section_header = line
            current_section_bullets = []
        elif stripped:
            parsed = parse_playbook_line(line)
            if parsed:
                current_section_bullets.append((parsed, line))
        # Empty lines are skipped for budget calculation

    # Don't forget the last section
    if current_section_header is not None:
        sections.append((current_section_header, current_section_bullets))

    # Collect all bullets with their metadata for prioritization
    all_bullets: list[tuple[ParsedBullet, str, str]] = []  # (parsed, line, section_header)
    for section_header, bullets in sections:
        for parsed, line in bullets:
            all_bullets.append((parsed, line, section_header))

    # Separate fresh bullets (0 interactions) from proven bullets
    # Fresh bullets get priority to ensure they can be evaluated at least once
    fresh_bullets = [(p, l, s) for p, l, s in all_bullets if _is_fresh_bullet(p)]
    proven_bullets = [(p, l, s) for p, l, s in all_bullets if not _is_fresh_bullet(p)]

    # Sort proven bullets by priority (highest first)
    proven_bullets.sort(key=lambda x: _compute_bullet_priority(x[0]), reverse=True)

    # Calculate token budget for bullets (reserve space for headers)
    section_headers = [s[0] for s in sections]
    header_tokens = count_tokens_approximate("\n".join(section_headers))
    available_for_bullets = token_budget - header_tokens - reserve_tokens

    if available_for_bullets <= 0:
        # Not enough budget even for headers, return just headers
        return "\n\n".join(section_headers)

    # Select bullets that fit within budget
    # IMPORTANT: Include fresh bullets first to ensure they survive at least one round
    selected_bullets: set[str] = set()  # Track by bullet ID
    used_tokens = 0

    # First pass: include fresh bullets (newly curated, need a chance to prove themselves)
    for parsed, line, _ in fresh_bullets:
        line_tokens = count_tokens_approximate(line)
        if used_tokens + line_tokens <= available_for_bullets:
            selected_bullets.add(parsed.id)
            used_tokens += line_tokens
        # If budget exhausted by fresh bullets alone, stop
        # (rare edge case - too many fresh bullets)

    # Second pass: fill remaining budget with proven bullets by priority
    for parsed, line, _ in proven_bullets:
        line_tokens = count_tokens_approximate(line)
        if used_tokens + line_tokens <= available_for_bullets:
            selected_bullets.add(parsed.id)
            used_tokens += line_tokens
        # Once budget exhausted, stop adding

    # Rebuild playbook preserving section order, only including selected bullets
    result_lines: list[str] = []
    for section_header, bullets in sections:
        result_lines.append(section_header)
        for parsed, line in bullets:
            if parsed.id in selected_bullets:
                result_lines.append(line)

    return "\n".join(result_lines)
