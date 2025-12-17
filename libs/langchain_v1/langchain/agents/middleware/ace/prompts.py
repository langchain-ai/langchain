"""Prompt templates for ACE middleware.

These prompts are adapted from the ACE framework for use within the
LangChain middleware context.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

# System prompt enhancement template
PLAYBOOK_INJECTION_TEMPLATE = """{original_prompt}

---

## ACE PLAYBOOK

The following playbook contains accumulated strategies, insights, and patterns
learned from previous interactions. Use relevant entries to improve your responses.
Each entry has helpful/harmful counts indicating its effectiveness.

{playbook}

---

{reflection_section}"""

REFLECTION_SECTION_TEMPLATE = """## PREVIOUS REFLECTION

The following reflection provides feedback on a previous attempt at this task:

{reflection}

Please consider this feedback to avoid repeating mistakes."""


# Reflector prompt for analyzing responses
REFLECTOR_PROMPT = """You are an expert analyst evaluating an AI agent's response.
Your job is to analyze whether the response was effective and which strategies helped or hurt.

## Instructions

1. Analyze the agent's reasoning and response quality
2. Identify what worked well and what could be improved
3. Tag each referenced playbook bullet as 'helpful', 'harmful', or 'neutral'
4. Provide a concise reflection with actionable insights

## Agent's Response

**User Query:**
{question}

**Agent's Reasoning/Response:**
{reasoning_trace}

**Feedback/Outcome:**
{feedback}

**Playbook Bullets Referenced:**
{bullets_used}

## Your Analysis

Respond in this exact JSON format:
{{
  "analysis": "Brief analysis of the response quality",
  "what_worked": "What strategies or approaches were effective",
  "what_failed": "What could be improved",
  "key_insight": "One actionable insight to remember for future similar tasks",
  "bullet_tags": [
    {{"id": "xxx-00001", "tag": "helpful"}},
    {{"id": "xxx-00002", "tag": "harmful"}}
  ]
}}

Only include bullets in bullet_tags that were actually referenced. Tag options:
- "helpful": The bullet contributed positively to the response
- "harmful": The bullet led to errors or poor reasoning
- "neutral": The bullet had no significant impact"""


# Curator prompt for updating the playbook
CURATOR_PROMPT = """You are a curator responsible for maintaining and improving \
a knowledge playbook.
Based on recent interactions and reflections, decide what updates to make.

## Current State

**Training Progress:** Step {current_step} of {total_samples}
**Token Budget:** {token_budget} tokens
**Playbook Statistics:**
{playbook_stats}

## Recent Reflection

{recent_reflection}

## Current Playbook

{current_playbook}

## Instructions

Analyze the reflection and decide what operations to perform:
- ADD: Add new insights or strategies that would be helpful
- UPDATE: Modify existing bullets to be more accurate (not implemented yet)
- DELETE: Remove bullets that are consistently harmful (not implemented yet)

Focus on:
1. Capturing new insights from the reflection
2. Not duplicating existing knowledge
3. Keeping bullets concise and actionable
4. Staying within the token budget

## Your Response

Respond in this exact JSON format:
{{
  "reasoning": "Your analysis of what updates are needed",
  "operations": [
    {{
      "type": "ADD",
      "section": "STRATEGIES & INSIGHTS",
      "content": "The new insight or strategy to add",
      "reason": "Why this is worth adding"
    }}
  ]
}}

Valid sections: STRATEGIES & INSIGHTS, FORMULAS & CALCULATIONS, CODE SNIPPETS & TEMPLATES,
COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS

If no updates are needed, return an empty operations list."""


def build_system_prompt_with_playbook(
    original_prompt: SystemMessage | str | None,
    playbook: str,
    reflection: str = "",
) -> str:
    """Build an enhanced system prompt that includes the ACE playbook.

    Args:
        original_prompt: The original system prompt/message.
        playbook: The current playbook content.
        reflection: Optional reflection from previous attempt.

    Returns:
        Enhanced system prompt with playbook injected.
    """
    # Extract text from SystemMessage if needed
    if original_prompt is None:
        original_text = "You are a helpful AI assistant."
    elif isinstance(original_prompt, SystemMessage):
        original_text = original_prompt.content
        if isinstance(original_text, list):
            # Handle multi-part content
            original_text = " ".join(
                part if isinstance(part, str) else str(part) for part in original_text
            )
    else:
        original_text = str(original_prompt)

    # Build reflection section if provided
    if reflection and reflection.strip() and reflection != "(empty)":
        reflection_section = REFLECTION_SECTION_TEMPLATE.format(reflection=reflection)
    else:
        reflection_section = ""

    return PLAYBOOK_INJECTION_TEMPLATE.format(
        original_prompt=original_text,
        playbook=playbook,
        reflection_section=reflection_section,
    )


def build_reflector_prompt(
    question: str,
    reasoning_trace: str,
    feedback: str,
    bullets_used: str,
) -> str:
    """Build the reflector prompt for analyzing a response.

    Args:
        question: The original user question/query.
        reasoning_trace: The agent's response/reasoning.
        feedback: Feedback about the response (success/failure info).
        bullets_used: Formatted string of playbook bullets that were referenced.

    Returns:
        Formatted reflector prompt.
    """
    return REFLECTOR_PROMPT.format(
        question=question,
        reasoning_trace=reasoning_trace,
        feedback=feedback,
        bullets_used=bullets_used,
    )


def build_curator_prompt(
    current_step: int,
    total_samples: int,
    token_budget: int,
    playbook_stats: str,
    recent_reflection: str,
    current_playbook: str,
) -> str:
    """Build the curator prompt for updating the playbook.

    Args:
        current_step: Current interaction count.
        total_samples: Expected total interactions (can be estimate).
        token_budget: Maximum tokens for playbook.
        playbook_stats: JSON string of playbook statistics.
        recent_reflection: Most recent reflection content.
        current_playbook: Current playbook content.

    Returns:
        Formatted curator prompt.
    """
    return CURATOR_PROMPT.format(
        current_step=current_step,
        total_samples=total_samples,
        token_budget=token_budget,
        playbook_stats=playbook_stats,
        recent_reflection=recent_reflection,
        current_playbook=current_playbook,
    )

