from langchain.prompts.prompt import PromptTemplate

_CREATE_DRAFT_ACTION_TEMPLATE_ = """
    This is the situation: {situation}
    These are the set of valid actions to take: {valid_actions}
    {call_to_action}
    """
CREATE_DRAFT_ACTION_PROMPT = PromptTemplate(
    input_variables=["situation", "valid_actions", "call_to_action"],
    template=_CREATE_DRAFT_ACTION_TEMPLATE_,
)

_CHECK_ACTION_VALIDITY_TEMPLATE_ = """
    Given the situation: {situation}
    And the action you chose: {initial_action}
    Is the action you in this set of valid actions: {valid_actions}?
    If not, choose the best valid action to take. If so, please return the original action
    """
CHECK_ACTION_VALIDITY_PROMPT = PromptTemplate(
    input_variables=["situation", "initial_action", "valid_actions"],
    template=_CHECK_ACTION_VALIDITY_TEMPLATE_,
)

_CHECK_ACTION_FORMAT_TEMPLATE_ = """
    This is the correct format for an action: {action_format}
    This is the chosen action: {validated_action}
    Convert the chosen action to the correct format.
    """
CHECK_ACTION_FORMAT_PROMPT = PromptTemplate(
    input_variables=["action_format", "validated_action"],
    template=_CHECK_ACTION_FORMAT_TEMPLATE_,
)

_CHECK_FORMAT_VALIDITY_TEMPLATE_ = """
    This is the correct format for an action: {action_format}
    This is a formatted action: {initial_format_validated_action}
    Return the action in the correct format.
"""

CHECK_FORMAT_VALIDITY_PROMPT = PromptTemplate(
    input_variables=["action_format", "initial_format_validated_action"],
    template=_CHECK_FORMAT_VALIDITY_TEMPLATE_,
)
