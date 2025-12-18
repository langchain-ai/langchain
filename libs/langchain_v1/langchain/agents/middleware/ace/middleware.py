"""ACE (Agentic Context Engineering) middleware implementation.

This middleware enables agents to self-improve by maintaining an evolving
playbook of strategies and insights learned from interactions.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
)
from langchain.chat_models import init_chat_model

from .playbook import (
    ACEPlaybook,
    add_bullet_to_playbook,
    extract_bullet_ids,
    extract_bullet_ids_from_comment,
    extract_playbook_bullets,
    get_max_bullet_id,
    get_playbook_stats,
    initialize_empty_playbook,
    prune_harmful_bullets,
    update_bullet_counts,
)
from .prompts import (
    build_curator_prompt,
    build_reflector_prompt,
    build_system_prompt_with_playbook,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# Private state annotation
_PrivateState = OmitFromSchema(input=True, output=True)


class ACEState(AgentState):
    """Extended agent state with ACE playbook tracking.

    This state schema adds fields for tracking the evolving playbook,
    reflections, and interaction counts used by ACE middleware.
    """

    # The evolving playbook
    ace_playbook: NotRequired[Annotated[dict[str, Any], _PrivateState]]

    # Last reflection content for context
    ace_last_reflection: NotRequired[Annotated[str, _PrivateState]]

    # Interaction counter for curator frequency
    ace_interaction_count: NotRequired[Annotated[int, _PrivateState]]


class ACEMiddleware(AgentMiddleware[ACEState, Any]):
    """ACE (Agentic Context Engineering) middleware for self-improving agents.

    This middleware implements the ACE framework which enables agents to
    self-improve by treating contexts as evolving playbooks that accumulate,
    refine, and organize strategies through a modular process of generation,
    reflection, and curation.

    The middleware operates in three phases:
    1. **Before model call**: Injects the playbook into the system prompt
    2. **After model call**: Analyzes the response with a reflector model
    3. **Periodically**: Curates the playbook to add new insights

    Attributes:
        state_schema: The extended state schema with ACE fields.
        tools: Empty list (no additional tools registered).

    Example:
        Basic usage with default settings:

        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ACEMiddleware

        ace = ACEMiddleware(reflector_model="gpt-4o-mini")

        agent = create_agent(
            model="gpt-4o",
            tools=[...],
            middleware=[ace],
        )
        ```

        With custom configuration:

        ```python
        ace = ACEMiddleware(
            reflector_model="gpt-4o-mini",
            curator_model="gpt-4o-mini",
            curator_frequency=10,
            initial_playbook=\"\"\"## STRATEGIES & INSIGHTS
        [str-00001] helpful=5 harmful=0 :: Always verify data types
        \"\"\",
            auto_prune=True,
            prune_threshold=0.6,
        )
        ```
    """

    state_schema = ACEState
    tools: list = []

    def __init__(
        self,
        *,
        reflector_model: str | BaseChatModel | None = None,
        curator_model: str | BaseChatModel | None = None,
        initial_playbook: str | None = None,
        curator_frequency: int = 5,
        playbook_token_budget: int = 80000,
        enable_reflection: bool = True,
        enable_curation: bool = True,
        auto_prune: bool = False,
        prune_threshold: float = 0.5,
        prune_min_interactions: int = 3,
    ) -> None:
        """Initialize ACE middleware.

        Args:
            reflector_model: Model for analyzing responses and tagging bullets.
                Can be a model name string or a `BaseChatModel` instance.
                If not provided, reflection is disabled.
            curator_model: Model for curating the playbook.
                If not provided, uses the reflector model.
            initial_playbook: Starting playbook content.
                If not provided, uses an empty template with standard sections.
            curator_frequency: Run curator every N interactions.
            playbook_token_budget: Maximum tokens for playbook content.
            enable_reflection: Whether to run reflector after model calls.
                Requires `reflector_model` to be set.
            enable_curation: Whether to periodically curate the playbook.
                Requires `curator_model` or `reflector_model` to be set.
            auto_prune: Whether to automatically prune harmful bullets.
            prune_threshold: Harmful ratio threshold for pruning (0-1).
            prune_min_interactions: Minimum interactions before pruning.
        """
        super().__init__()

        # Store model specs (initialize lazily)
        self._reflector_model_spec = reflector_model
        self._curator_model_spec = curator_model or reflector_model
        self._reflector_model: BaseChatModel | None = None
        self._curator_model: BaseChatModel | None = None

        # Configuration
        self.initial_playbook = initial_playbook or initialize_empty_playbook()
        self.curator_frequency = curator_frequency
        self.playbook_token_budget = playbook_token_budget
        self.enable_reflection = enable_reflection and reflector_model is not None
        self.enable_curation = enable_curation and (
            curator_model is not None or reflector_model is not None
        )
        self.auto_prune = auto_prune
        self.prune_threshold = prune_threshold
        self.prune_min_interactions = prune_min_interactions

    def _get_reflector_model(self) -> BaseChatModel | None:
        """Get or initialize the reflector model."""
        if self._reflector_model is not None:
            return self._reflector_model

        if self._reflector_model_spec is None:
            return None

        if isinstance(self._reflector_model_spec, str):
            self._reflector_model = init_chat_model(self._reflector_model_spec)
        else:
            self._reflector_model = self._reflector_model_spec

        return self._reflector_model

    def _get_curator_model(self) -> BaseChatModel | None:
        """Get or initialize the curator model."""
        if self._curator_model is not None:
            return self._curator_model

        if self._curator_model_spec is None:
            return None

        if isinstance(self._curator_model_spec, str):
            self._curator_model = init_chat_model(self._curator_model_spec)
        else:
            self._curator_model = self._curator_model_spec

        return self._curator_model

    def _get_playbook(self, state: AgentState[Any]) -> ACEPlaybook:
        """Get playbook from state or create default."""
        playbook_data = state.get("ace_playbook")
        if playbook_data:
            return ACEPlaybook.from_dict(playbook_data)
        # Scan initial playbook for existing bullet IDs to avoid duplicates
        max_id = get_max_bullet_id(self.initial_playbook)
        return ACEPlaybook(
            content=self.initial_playbook,
            next_global_id=max_id + 1,
            stats=get_playbook_stats(self.initial_playbook),
        )

    def _get_last_user_message(self, messages: Sequence[BaseMessage]) -> str:
        """Extract the last user message from the conversation."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        part if isinstance(part, str) else str(part) for part in content
                    )
        return ""

    def _extract_tool_feedback(self, messages: Sequence[BaseMessage]) -> str:
        """Extract tool results and errors from messages for reflector feedback.

        Scans recent messages for tool calls and their results, providing
        valuable context for the reflector to understand what worked or failed.

        Args:
            messages: List of messages from the conversation.

        Returns:
            Formatted string describing tool results and any errors.
        """
        tool_results: list[str] = []
        errors: list[str] = []

        for msg in messages:
            # Check for tool call results
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown_tool")
                content = msg.content if isinstance(msg.content, str) else str(msg.content)

                # Check for errors in tool results
                if "error" in content.lower() or "exception" in content.lower():
                    errors.append(f"Tool '{tool_name}' error: {content[:200]}")
                else:
                    # Truncate long results
                    max_preview_len = 150
                    if len(content) > max_preview_len:
                        result_preview = content[:max_preview_len] + "..."
                    else:
                        result_preview = content
                    tool_results.append(f"Tool '{tool_name}': {result_preview}")

            # Check for tool calls in AI messages (to understand what was attempted)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    args_str = str(tool_args)[:100]
                    tool_results.append(f"Called '{tool_name}' with: {args_str}")

        # Build feedback string
        parts = []
        if errors:
            parts.append("**Errors encountered:**\n" + "\n".join(f"- {e}" for e in errors))
        if tool_results:
            parts.append("**Tool usage:**\n" + "\n".join(f"- {r}" for r in tool_results[-5:]))

        if parts:
            return "\n\n".join(parts)
        return "Response generated successfully (no tools used)"

    def _extract_json_from_response(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from model response text."""
        # Try direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        import re

        json_pattern = r"```(?:json)?\s*(.*?)\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try to find raw JSON object
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    def _build_reflection_summary(self, parsed: dict[str, Any]) -> str:
        """Build a comprehensive reflection summary from parsed reflector output.

        Args:
            parsed: Parsed JSON from reflector response.

        Returns:
            Formatted reflection string for display and storage.
        """
        parts = []

        # Error analysis (if present)
        error_id = parsed.get("error_identification", "")
        if error_id and error_id != "N/A - response was correct":
            parts.append(f"**Error:** {error_id}")

            root_cause = parsed.get("root_cause_analysis", "")
            if root_cause and root_cause != "N/A - response was correct":
                parts.append(f"**Root Cause:** {root_cause}")

            correct_approach = parsed.get("correct_approach", "")
            if correct_approach and correct_approach != "N/A - response was correct":
                parts.append(f"**Correct Approach:** {correct_approach}")

        # Key insight (always included)
        key_insight = parsed.get("key_insight", "")
        if key_insight:
            parts.append(f"**Key Insight:** {key_insight}")

        # Bullet tags summary
        bullet_tags = parsed.get("bullet_tags", [])
        if bullet_tags:
            helpful = [t["id"] for t in bullet_tags if t.get("tag") == "helpful"]
            harmful = [t["id"] for t in bullet_tags if t.get("tag") == "harmful"]
            if helpful:
                parts.append(f"**Helpful bullets:** {', '.join(helpful)}")
            if harmful:
                parts.append(f"**Harmful bullets:** {', '.join(harmful)}")

        return "\n".join(parts) if parts else parsed.get("key_insight", "")

    @override
    def before_agent(self, state: ACEState, runtime: Runtime) -> dict[str, Any] | None:
        """Initialize ACE state at the start of agent execution."""
        if state.get("ace_playbook") is None:
            # Scan initial playbook for existing bullet IDs to avoid duplicates
            max_id = get_max_bullet_id(self.initial_playbook)
            playbook = ACEPlaybook(
                content=self.initial_playbook,
                next_global_id=max_id + 1,
                stats=get_playbook_stats(self.initial_playbook),
            )
            return {
                "ace_playbook": playbook.to_dict(),
                "ace_last_reflection": "",
                "ace_interaction_count": 0,
            }
        return None

    @override
    async def abefore_agent(self, state: ACEState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of before_agent."""
        return self.before_agent(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Inject playbook into system prompt before model call."""
        state = request.state
        playbook = self._get_playbook(state)
        last_reflection = cast("str", state.get("ace_last_reflection", ""))

        # Build enhanced system prompt
        enhanced_prompt = build_system_prompt_with_playbook(
            original_prompt=request.system_message,
            playbook=playbook.content,
            reflection=last_reflection,
        )

        # Override system message and call handler
        modified_request = request.override(system_message=SystemMessage(content=enhanced_prompt))

        return handler(modified_request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async version of wrap_model_call."""
        state = request.state
        playbook = self._get_playbook(state)
        last_reflection = cast("str", state.get("ace_last_reflection", ""))

        enhanced_prompt = build_system_prompt_with_playbook(
            original_prompt=request.system_message,
            playbook=playbook.content,
            reflection=last_reflection,
        )

        modified_request = request.override(system_message=SystemMessage(content=enhanced_prompt))

        return await handler(modified_request)

    @override
    def after_model(self, state: ACEState, runtime: Runtime) -> dict[str, Any] | None:
        """Analyze response and update playbook after model call."""
        if not self.enable_reflection:
            return self._increment_interaction_count(state)

        reflector = self._get_reflector_model()
        if reflector is None:
            return self._increment_interaction_count(state)

        messages = state.get("messages", [])
        if not messages:
            return self._increment_interaction_count(state)

        # Get latest AI message
        latest_msg = messages[-1]
        if not isinstance(latest_msg, AIMessage):
            return self._increment_interaction_count(state)

        # Get playbook and extract bullet IDs
        playbook = self._get_playbook(state)
        ai_content = (
            latest_msg.content if isinstance(latest_msg.content, str) else str(latest_msg.content)
        )
        # Try comment-based extraction first, fall back to inline citations
        bullet_ids = extract_bullet_ids_from_comment(ai_content)
        if not bullet_ids:
            bullet_ids = extract_bullet_ids(ai_content)
        bullets_used = extract_playbook_bullets(playbook.content, bullet_ids)

        # Build reflector prompt with tool feedback
        user_question = self._get_last_user_message(messages[:-1])
        tool_feedback = self._extract_tool_feedback(messages)
        reflector_prompt = build_reflector_prompt(
            question=user_question,
            reasoning_trace=ai_content,
            feedback=tool_feedback,
            bullets_used=bullets_used,
        )

        # Call reflector
        try:
            response = reflector.invoke(reflector_prompt)
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = self._extract_json_from_response(response_text)
        except Exception:
            return self._increment_interaction_count(state)

        if not parsed:
            return self._increment_interaction_count(state)

        # Update bullet counts
        bullet_tags = parsed.get("bullet_tags", [])
        updated_content = update_bullet_counts(playbook.content, bullet_tags)

        # Build comprehensive reflection from parsed fields
        reflection = self._build_reflection_summary(parsed)

        # Auto-prune if enabled
        if self.auto_prune:
            updated_content = prune_harmful_bullets(
                updated_content,
                threshold=self.prune_threshold,
                min_interactions=self.prune_min_interactions,
            )

        # Increment interaction count
        interaction_count = state.get("ace_interaction_count", 0) + 1

        # Prepare base updates
        updated_playbook = ACEPlaybook(
            content=updated_content,
            next_global_id=playbook.next_global_id,
            stats=get_playbook_stats(updated_content),
        )

        updates: dict[str, Any] = {
            "ace_playbook": updated_playbook.to_dict(),
            "ace_last_reflection": reflection,
            "ace_interaction_count": interaction_count,
        }

        # Run curator if threshold reached
        if self.enable_curation and interaction_count % self.curator_frequency == 0:
            curated = self._run_curator(updated_playbook, reflection)
            if curated:
                updates["ace_playbook"] = curated.to_dict()

        return updates

    @override
    async def aafter_model(self, state: ACEState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of after_model."""
        if not self.enable_reflection:
            return self._increment_interaction_count(state)

        reflector = self._get_reflector_model()
        if reflector is None:
            return self._increment_interaction_count(state)

        messages = state.get("messages", [])
        if not messages:
            return self._increment_interaction_count(state)

        latest_msg = messages[-1]
        if not isinstance(latest_msg, AIMessage):
            return self._increment_interaction_count(state)

        playbook = self._get_playbook(state)
        ai_content = (
            latest_msg.content if isinstance(latest_msg.content, str) else str(latest_msg.content)
        )
        # Try comment-based extraction first, fall back to inline citations
        bullet_ids = extract_bullet_ids_from_comment(ai_content)
        if not bullet_ids:
            bullet_ids = extract_bullet_ids(ai_content)
        bullets_used = extract_playbook_bullets(playbook.content, bullet_ids)

        user_question = self._get_last_user_message(messages[:-1])
        tool_feedback = self._extract_tool_feedback(messages)
        reflector_prompt = build_reflector_prompt(
            question=user_question,
            reasoning_trace=ai_content,
            feedback=tool_feedback,
            bullets_used=bullets_used,
        )

        try:
            response = await reflector.ainvoke(reflector_prompt)
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = self._extract_json_from_response(response_text)
        except Exception:
            return self._increment_interaction_count(state)

        if not parsed:
            return self._increment_interaction_count(state)

        bullet_tags = parsed.get("bullet_tags", [])
        updated_content = update_bullet_counts(playbook.content, bullet_tags)
        reflection = self._build_reflection_summary(parsed)

        if self.auto_prune:
            updated_content = prune_harmful_bullets(
                updated_content,
                threshold=self.prune_threshold,
                min_interactions=self.prune_min_interactions,
            )

        interaction_count = state.get("ace_interaction_count", 0) + 1

        updated_playbook = ACEPlaybook(
            content=updated_content,
            next_global_id=playbook.next_global_id,
            stats=get_playbook_stats(updated_content),
        )

        updates: dict[str, Any] = {
            "ace_playbook": updated_playbook.to_dict(),
            "ace_last_reflection": reflection,
            "ace_interaction_count": interaction_count,
        }

        if self.enable_curation and interaction_count % self.curator_frequency == 0:
            curated = await self._arun_curator(updated_playbook, reflection)
            if curated:
                updates["ace_playbook"] = curated.to_dict()

        return updates

    def _increment_interaction_count(self, state: ACEState) -> dict[str, Any]:
        """Helper to just increment interaction count."""
        return {
            "ace_interaction_count": state.get("ace_interaction_count", 0) + 1,
        }

    def _run_curator(self, playbook: ACEPlaybook, reflection: str) -> ACEPlaybook | None:
        """Run the curator to update the playbook with new insights."""
        curator = self._get_curator_model()
        if curator is None:
            return None

        curator_prompt = build_curator_prompt(
            current_step=1,  # Could track this in state
            total_samples=100,  # Estimate
            token_budget=self.playbook_token_budget,
            playbook_stats=json.dumps(playbook.stats, indent=2),
            recent_reflection=reflection,
            current_playbook=playbook.content,
        )

        try:
            response = curator.invoke(curator_prompt)
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = self._extract_json_from_response(response_text)
        except Exception:
            return None

        if not parsed:
            return None

        operations = parsed.get("operations", [])
        updated_content = playbook.content
        next_id = playbook.next_global_id

        for op in operations:
            if op.get("type") == "ADD":
                section = op.get("section", "OTHERS")
                content = op.get("content", "")
                if content:
                    updated_content, next_id = add_bullet_to_playbook(
                        updated_content, section, content, next_id
                    )

        return ACEPlaybook(
            content=updated_content,
            next_global_id=next_id,
            stats=get_playbook_stats(updated_content),
        )

    async def _arun_curator(self, playbook: ACEPlaybook, reflection: str) -> ACEPlaybook | None:
        """Async version of curator."""
        curator = self._get_curator_model()
        if curator is None:
            return None

        curator_prompt = build_curator_prompt(
            current_step=1,
            total_samples=100,
            token_budget=self.playbook_token_budget,
            playbook_stats=json.dumps(playbook.stats, indent=2),
            recent_reflection=reflection,
            current_playbook=playbook.content,
        )

        try:
            response = await curator.ainvoke(curator_prompt)
            response_text = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = self._extract_json_from_response(response_text)
        except Exception:
            return None

        if not parsed:
            return None

        operations = parsed.get("operations", [])
        updated_content = playbook.content
        next_id = playbook.next_global_id

        for op in operations:
            if op.get("type") == "ADD":
                section = op.get("section", "OTHERS")
                content = op.get("content", "")
                if content:
                    updated_content, next_id = add_bullet_to_playbook(
                        updated_content, section, content, next_id
                    )

        return ACEPlaybook(
            content=updated_content,
            next_global_id=next_id,
            stats=get_playbook_stats(updated_content),
        )
