"""Prompt for e-SNLI."""
# From https://arxiv.org/pdf/2203.11171.pdf

from langchain.prompt import Prompt

_PROMPT_TEMPLATE = """Premise:
"A person on a horse jumps over a broken down airplane."
Based on this premise, can we conclude the hypothesis "A person is training his horse for a competition." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: The person is not necessarily training his horse. The answer is it is not possible to tell.
Premise:
"A person on a horse jumps over a broken down airplane."
Based on this premise, can we conclude the hypothesis "A person is at a diner, ordering an omelette." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: One jumping horse cannot be in a diner ordering food. The answer is no.
Premise:
"A person on a horse jumps over a broken down airplane."
Based on this premise, can we conclude the hypothesis "A person is outdoors, on a horse." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: A broken down airplane is outdoors. The answer is yes.
Premise:
"Children smiling and waving at camera."
Based on this premise, can we conclude the hypothesis "They are smiling at their parents." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind
it. The answer is it is not possible to tell.
Premise:
"Children smiling and waving at camera."
Based on this premise, can we conclude the hypothesis "The kids are frowning." is true? OPTIONS:
- yes
- no
- it is not possible to tell
A: One cannot be smiling and frowning at the same time. The answer is no.
Premise:
"Children smiling and waving at camera."
Based on this premise, can we conclude the hypothesis "There are children present." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A:The children must be present to see them smiling and waving. The answer is yes.
Premise:
\"{premise}\"
Based on this premise, can we conclude the hypothesis \"{hypothesis}\" is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: """

ESNLI_PROMPT = Prompt(
    input_variables=["premise", "hypothesis"],
    template=_PROMPT_TEMPLATE,
)
