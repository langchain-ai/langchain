"""PII redaction middleware for agents.

This middleware detects and redacts personally identifiable information (PII)
from messages before they are sent to model providers, and restores original
values in model responses for tool execution.
"""

import json
import re
import uuid
from re import Pattern
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest


class PIIRedactionConfig(TypedDict):
    """Configuration for PII redaction middleware."""

    rules: NotRequired[dict[str, Pattern[str]]]
    """A record of PII detection rules to apply. Maps rule names to regex patterns."""


class PIIRedactionMiddleware(AgentMiddleware):
    r"""Middleware that detects and redacts personally identifiable information (PII).

    This middleware intercepts agent execution at two points:

    ### Request Phase (`modify_model_request`)
    - Applies regex-based pattern matching to all message content
    - Processes both message text and AIMessage tool call arguments
    - Each matched pattern generates:
      - Unique identifier: `generate_redaction_id()` → `"abc123"`
      - Redaction marker: `[REDACTED_{RULE_NAME}_{ID}]` → `"[REDACTED_SSN_abc123]"`
      - Redaction map entry: `{ "abc123": "123-45-6789" }`
    - Returns modified request with redacted message content

    ### Response Phase (`after_model`)
    - Scans AIMessage responses for redaction markers
    - Replaces markers with original values from redaction map
    - Handles both standard responses and structured output
    - Returns new message instances to update state

    ## Data Flow

    ```
    User Input: "My SSN is 123-45-6789"
        ↓ [modify_model_request]
    Model Request: "My SSN is [REDACTED_SSN_abc123]"
        ↓ [model invocation]
    Model Response: tool_call({ "ssn": "[REDACTED_SSN_abc123]" })
        ↓ [after_model]
    Tool Execution: tool({ "ssn": "123-45-6789" })
    ```

    ## Limitations

    This middleware provides model provider isolation only. PII may still be present in:
    - LangGraph state checkpoints (memory, databases)
    - Network traffic between client and application server
    - Application logs and trace data
    - Tool execution arguments and responses
    - Final agent output

    For comprehensive PII protection, implement additional controls at the application,
    network, and storage layers.

    Example:
        ```python
        import re
        from langchain.agents.middleware.pii_redaction import PIIRedactionMiddleware
        from langchain.agents import create_agent

        PII_RULES = {
            "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        }

        agent = create_agent(
            model="openai:gpt-4",
            tools=[some_tool],
            middleware=[PIIRedactionMiddleware(rules=PII_RULES)],
        )

        result = await agent.invoke({"messages": [HumanMessage("Look up SSN 123-45-6789")]})
        # Model request: "Look up SSN [REDACTED_SSN_abc123]"
        # Model response: tool_call({ "ssn": "[REDACTED_SSN_abc123]" })
        # Tool receives: { "ssn": "123-45-6789" }
        ```
    """

    def __init__(self, rules: dict[str, Pattern[str]] | None = None) -> None:
        """Initialize the PII redaction middleware.

        Args:
            rules: Record of detection rules mapping rule names to regex patterns.
                Rule names are normalized to uppercase and used in redaction markers.
                Patterns should be compiled regex objects for better performance.
        """
        super().__init__()
        self.rules = rules or {}
        self.redaction_map: dict[str, str] = {}

    def _generate_redaction_id(self) -> str:
        """Generate a unique ID for a redaction."""
        return str(uuid.uuid4())[:8]

    def _apply_pii_rules(
        self, text: str, rules: dict[str, Pattern[str]], redaction_map: dict[str, str]
    ) -> str:
        """Apply PII detection rules to text with ID tracking.

        Args:
            text: The text to process for PII.
            rules: Dictionary of rule names to regex patterns.
            redaction_map: Dictionary to store original values by ID.

        Returns:
            Text with PII replaced by redaction markers.
        """
        processed_text = text

        for name, pattern in rules.items():
            # Normalize rule name for redaction marker
            replacement = re.sub(r"[^a-zA-Z0-9_-]", "", name.upper())

            def replace_match(match: re.Match[str]) -> str:
                original_value = match.group(0)
                redaction_id = self._generate_redaction_id()
                redaction_map[redaction_id] = original_value
                return f"[REDACTED_{replacement}_{redaction_id}]"  # noqa: B023

            processed_text = pattern.sub(replace_match, processed_text)

        return processed_text

    def _process_message(
        self, message: AnyMessage, rules: dict[str, Pattern[str]], redaction_map: dict[str, str]
    ) -> AnyMessage:
        """Process a single message for PII detection and redaction.

        Args:
            message: The message to process.
            rules: Dictionary of rule names to regex patterns.
            redaction_map: Dictionary to store original values by ID.

        Returns:
            New message instance with redacted content if changes were made.

        Raises:
            ValueError: If the message type is not supported.
        """
        # Handle basic message types (HumanMessage, ToolMessage, SystemMessage)
        if isinstance(message, (HumanMessage, ToolMessage, SystemMessage)):
            content = message.content
            if isinstance(content, str):
                processed_content = self._apply_pii_rules(content, rules, redaction_map)

                if processed_content != content:
                    # Create new message instance with redacted content
                    # For ToolMessage, we need to preserve tool_call_id
                    if isinstance(message, ToolMessage):
                        return message.__class__(
                            content=processed_content,
                            tool_call_id=message.tool_call_id,
                            **message.additional_kwargs,
                        )
                    return message.__class__(content=processed_content, **message.additional_kwargs)

            return message

        # Handle AI messages
        if isinstance(message, AIMessage):
            content_changed = False
            tool_calls_changed = False

            # Process content
            if isinstance(message.content, str):
                processed_content = self._apply_pii_rules(message.content, rules, redaction_map)
                if processed_content != message.content:
                    content_changed = True
                    new_content = processed_content
                else:
                    new_content = message.content
            else:
                # Handle non-string content by converting to JSON
                content_str = json.dumps(message.content)
                processed_content_str = self._apply_pii_rules(content_str, rules, redaction_map)
                if processed_content_str != content_str:
                    content_changed = True
                    new_content = json.loads(processed_content_str)
                else:
                    new_content = message.content

            # Process tool calls
            new_tool_calls = message.tool_calls
            if message.tool_calls:
                tool_calls_str = json.dumps(message.tool_calls)
                processed_tool_calls_str = self._apply_pii_rules(
                    tool_calls_str, rules, redaction_map
                )
                if processed_tool_calls_str != tool_calls_str:
                    tool_calls_changed = True
                    new_tool_calls = json.loads(processed_tool_calls_str)

            if content_changed or tool_calls_changed:
                return AIMessage(
                    content=new_content, tool_calls=new_tool_calls, **message.additional_kwargs
                )

            return message

        msg = f"Unsupported message type: {type(message)}"
        raise ValueError(msg)

    def _restore_redacted_values(self, text: str, redaction_map: dict[str, str]) -> str:
        """Restore original values from redacted text using the redaction map.

        Args:
            text: The text containing redaction markers.
            redaction_map: Dictionary mapping redaction IDs to original values.

        Returns:
            Text with redaction markers replaced by original values.
        """
        # Pattern to match redacted values like [REDACTED_SSN_abc123]
        redaction_pattern = re.compile(r"\[REDACTED_[A-Z_]+_(\w+)\]")

        def replace_redaction(match: re.Match[str]) -> str:
            redaction_id = match.group(1)
            if redaction_id in redaction_map:
                return redaction_map[redaction_id]
            return match.group(0)  # Keep original if no mapping found

        return redaction_pattern.sub(replace_redaction, text)

    def _restore_message(
        self, message: AnyMessage, redaction_map: dict[str, str]
    ) -> tuple[AnyMessage, bool]:
        """Restore redacted values in a message.

        Args:
            message: The message to restore.
            redaction_map: Dictionary mapping redaction IDs to original values.

        Returns:
            Tuple of (restored_message, changed) where changed indicates if any
            redactions were restored.

        Raises:
            ValueError: If the message type is not supported.
        """
        # Handle basic message types
        if isinstance(message, (HumanMessage, ToolMessage, SystemMessage)):
            content = message.content
            if isinstance(content, str):
                restored_content = self._restore_redacted_values(content, redaction_map)
                if restored_content != content:
                    # For ToolMessage, we need to preserve tool_call_id
                    if isinstance(message, ToolMessage):
                        new_message = message.__class__(
                            content=restored_content,
                            tool_call_id=message.tool_call_id,
                            **message.additional_kwargs,
                        )
                    else:
                        new_message = message.__class__(
                            content=restored_content, **message.additional_kwargs
                        )
                    return new_message, True

            return message, False

        # Handle AI messages
        if isinstance(message, AIMessage):
            content_changed = False
            tool_calls_changed = False

            # Restore content
            if isinstance(message.content, str):
                restored_content = self._restore_redacted_values(message.content, redaction_map)
                if restored_content != message.content:
                    content_changed = True
                    new_content = restored_content
                else:
                    new_content = message.content
            else:
                # Handle non-string content by converting to JSON
                content_str = json.dumps(message.content)
                restored_content_str = self._restore_redacted_values(content_str, redaction_map)
                if restored_content_str != content_str:
                    content_changed = True
                    new_content = json.loads(restored_content_str)
                else:
                    new_content = message.content

            # Restore tool calls
            new_tool_calls = message.tool_calls
            if message.tool_calls:
                tool_calls_str = json.dumps(message.tool_calls)
                restored_tool_calls_str = self._restore_redacted_values(
                    tool_calls_str, redaction_map
                )
                if restored_tool_calls_str != tool_calls_str:
                    tool_calls_changed = True
                    new_tool_calls = json.loads(restored_tool_calls_str)

            if content_changed or tool_calls_changed:
                new_message = AIMessage(
                    content=new_content, tool_calls=new_tool_calls, **message.additional_kwargs
                )
                return new_message, True

            return message, False

        msg = f"Unsupported message type: {type(message)}"
        raise ValueError(msg)

    def modify_model_request(
        self,
        request: ModelRequest,
        state: AgentState,  # noqa: ARG002
        runtime: Runtime,
    ) -> ModelRequest:
        """Modify model request to redact PII from messages.

        Args:
            request: The model request to modify.
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Modified model request with redacted messages.
        """
        # Get rules from context or use default rules
        try:
            context_config = getattr(runtime.context, "PIIRedactionMiddleware", {})
            rules = context_config.get("rules", self.rules)
        except AttributeError:
            # If context doesn't have the expected structure, use default rules
            rules = self.rules

        # If no rules are provided, skip processing
        if not rules:
            return request

        # Process all messages
        processed_messages = []
        for message in request.messages:
            processed_message = self._process_message(message, rules, self.redaction_map)
            processed_messages.append(processed_message)

        return ModelRequest(
            model=request.model,
            system_prompt=request.system_prompt,
            messages=processed_messages,
            tool_choice=request.tool_choice,
            tools=request.tools,
            response_format=request.response_format,
            model_settings=request.model_settings,
        )

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Restore redacted values in model responses.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            State updates with restored messages, or None if no changes needed.
        """
        # If no redactions were made, skip processing
        if not self.redaction_map:
            return None

        messages = state["messages"]
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Check if the last message is a structured response
        second_last_message = messages[-2] if len(messages) > 1 else None

        # Restore the last message
        restored_last_message, last_changed = self._restore_message(
            last_message, self.redaction_map
        )

        if not last_changed:
            return None

        # Check for structured response in the last message
        structured_response = None
        if (
            isinstance(last_message, AIMessage)
            and not last_message.tool_calls
            and isinstance(last_message.content, str)
            and last_message.content.startswith("{")
            and last_message.content.endswith("}")
        ):
            try:
                restored_content = self._restore_redacted_values(
                    last_message.content, self.redaction_map
                )
                structured_response = json.loads(restored_content)
            except (json.JSONDecodeError, ValueError):
                # Ignore JSON parsing errors
                pass

        # Check if the second last message is a structured response tool call
        is_structured_response_tool_call = (
            isinstance(second_last_message, AIMessage)
            and second_last_message.tool_calls
            and any(
                call.get("name", "").startswith("extract-")
                for call in second_last_message.tool_calls
            )
        )

        if is_structured_response_tool_call:
            restored_second_last_message, second_last_changed = self._restore_message(
                second_last_message, self.redaction_map
            )

            # Extract structured response from tool call
            if second_last_message.tool_calls:
                structured_tool_call = next(
                    (
                        call
                        for call in second_last_message.tool_calls
                        if call.get("name", "").startswith("extract-")
                    ),
                    None,
                )
                if structured_tool_call:
                    tool_args_str = json.dumps(structured_tool_call.get("args", {}))
                    restored_tool_args_str = self._restore_redacted_values(
                        tool_args_str, self.redaction_map
                    )
                    try:  # noqa: SIM105
                        structured_response = json.loads(restored_tool_args_str)
                    except (json.JSONDecodeError, ValueError):
                        pass

            if last_changed or second_last_changed:
                return {
                    **state,
                    **({"structured_response": structured_response} if structured_response else {}),
                    "messages": [
                        RemoveMessage(id=second_last_message.id),
                        RemoveMessage(id=last_message.id),
                        restored_second_last_message,
                        restored_last_message,
                    ],
                }

        return {
            **state,
            **({"structured_response": structured_response} if structured_response else {}),
            "messages": [
                RemoveMessage(id=last_message.id),
                restored_last_message,
            ],
        }
