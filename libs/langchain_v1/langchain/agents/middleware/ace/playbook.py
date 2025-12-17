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
from dataclasses import dataclass, field
from typing import Any

# Thresholds for classifying bullet performance
_HIGH_PERFORMING_HELPFUL_MIN = 5  # Minimum helpful count for high-performing bullets
_HIGH_PERFORMING_HARMFUL_MAX = 2  # Maximum harmful count for high-performing bullets

# Section slug mappings for bullet ID generation
_SECTION_SLUGS: dict[str, str] = {
    "strategies_and_insights": "str",
    "strategies & insights": "str",
    "formulas_and_calculations": "cal",
    "formulas & calculations": "cal",
    "code_snippets_and_templates": "cod",
    "code snippets & templates": "cod",
    "common_mistakes_to_avoid": "mis",
    "common mistakes to avoid": "mis",
    "problem-solving_heuristics": "heu",
    "problem-solving heuristics": "heu",
    "context_clues_and_indicators": "ctx",
    "context clues & indicators": "ctx",
    "others": "oth",
    "general": "gen",
}

_DEFAULT_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


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


def get_section_slug(section_name: str) -> str:
    """Get the 3-letter slug for a section name.

    Args:
        section_name: Section name (e.g., "strategies_and_insights").

    Returns:
        3-letter slug (e.g., "str").
    """
    normalized = section_name.lower().strip()
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


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into playbook line format.

    Args:
        bullet_id: Unique identifier for the bullet.
        helpful: Count of helpful occurrences.
        harmful: Count of harmful occurrences.
        content: The bullet content.

    Returns:
        Formatted playbook line.
    """
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def extract_bullet_ids(text: str) -> list[str]:
    """Extract bullet IDs referenced in text.

    Args:
        text: Text that may contain bullet ID references.

    Returns:
        List of bullet IDs found.
    """
    pattern = r"\[([a-z]{3}-\d{5})\]"
    return re.findall(pattern, text)


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
                    format_playbook_line(
                        parsed.id, parsed.helpful, parsed.harmful, parsed.content
                    )
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
    section: str,
    content: str,
    next_global_id: int,
) -> tuple[str, int]:
    """Add a new bullet to the playbook.

    Args:
        playbook_text: Current playbook content.
        section: Section name to add the bullet to.
        content: Content of the new bullet.
        next_global_id: Next available global ID.

    Returns:
        Tuple of (updated_playbook, new_next_global_id).
    """
    slug = get_section_slug(section)
    new_id = f"{slug}-{next_global_id:05d}"
    new_line = format_playbook_line(new_id, 0, 0, content)

    lines = playbook_text.strip().split("\n")
    new_lines: list[str] = []
    section_normalized = section.lower().replace(" ", "_").replace("&", "and")
    current_section: str | None = None
    added = False

    for line in lines:
        new_lines.append(line)

        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(" ", "_").replace("&", "and")

            # If we just left the target section, add the bullet
            if not added and current_section != section_normalized:
                # Check if previous section was target
                pass

        # Add bullet after section header if this is the target section
        if (
            not added
            and current_section == section_normalized
            and not line.strip().startswith("##")
        ):
            # Insert before this line if it's the start of content
            pass

    # Simple approach: find section and append after header
    result_lines: list[str] = []

    for line in lines:
        result_lines.append(line)

        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            normalized = header.lower().replace(" ", "_").replace("&", "and")
            if normalized == section_normalized:
                # Add the new bullet right after the section header
                result_lines.append(new_line)
                added = True

    if not added:
        # Section not found, add to OTHERS
        for i, line in enumerate(result_lines):
            if line.strip() == "## OTHERS":
                result_lines.insert(i + 1, new_line)
                added = True
                break

    if not added:
        # No OTHERS section, append at end
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

