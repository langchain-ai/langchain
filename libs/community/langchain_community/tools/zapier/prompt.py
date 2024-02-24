# flake8: noqa
BASE_ZAPIER_TOOL_PROMPT = (
    "A wrapper around Zapier NLA actions. "
    "The input to this tool is a natural language instruction, "
    'for example "get the latest email from my bank" or '
    '"send a slack message to the #general channel". '
    "Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. "
    "For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. "
    "Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. "
    "Do not make up params, they will be explicitly specified in the tool description. "
    "If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. "
    "If you get a none or null response, STOP EXECUTION, do not try to another tool!"
    "This tool specifically used for: {zapier_description}, "
    "and has params: {params}"
)
