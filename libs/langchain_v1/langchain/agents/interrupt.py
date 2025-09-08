"""Interrupt types to use with agent inbox like setups."""

from typing import Literal, Union

from typing_extensions import TypedDict


class HumanInterruptConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.

    Attributes:
        allow_ignore: Whether the human can choose to ignore/skip the current step
        allow_respond: Whether the human can provide a text response/feedback
        allow_edit: Whether the human can edit the provided content/state
        allow_accept: Whether the human can accept/approve the current state
    """

    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    """Represents a request for human action within the graph execution.

    Contains the action type and any associated arguments needed for the action.

    Attributes:
        action: The type or name of action being requested (e.g., "Approve XYZ action")
        args: Key-value pairs of arguments needed for the action
    """

    action: str
    args: dict


class HumanInterrupt(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    This is passed to the `interrupt` function when execution is paused for human input.

    Attributes:
        action_request: The specific action being requested from the human
        config: Configuration defining what actions are allowed
        description: Optional detailed description of what input is needed

    Example:
        ```python
        # Extract a tool call from the state and create an interrupt request
        request = HumanInterrupt(
            action_request=ActionRequest(
                action="run_command",  # The action being requested
                args={"command": "ls", "args": ["-l"]},  # Arguments for the action
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,  # Allow skipping this step
                allow_respond=True,  # Allow text feedback
                allow_edit=False,  # Don't allow editing
                allow_accept=True,  # Allow direct acceptance
            ),
            description="Please review the command before execution",
        )
        # Send the interrupt request and get the response
        response = interrupt([request])[0]
        ```
    """

    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None


class HumanResponse(TypedDict):
    """Human response.

    The response provided by a human to an interrupt,
    which is returned when graph execution resumes.

    Attributes:
        type: The type of response:

            - "accept": Approves the current state without changes
            - "ignore": Skips/ignores the current step
            - "response": Provides text feedback or instructions
            - "edit": Modifies the current state/content
        args: The response payload:

            - None: For ignore/accept actions
            - str: For text responses
            - ActionRequest: For edit actions with updated content
    """

    type: Literal["accept", "ignore", "response", "edit"]
    args: Union[None, str, ActionRequest]
