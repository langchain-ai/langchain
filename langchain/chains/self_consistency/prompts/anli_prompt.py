"""Prompts for adversarial NLI."""

# From https://arxiv.org/pdf/2203.11171.pdf

from langchain.prompt import Prompt


_PROMPT_TEMPLATE = """Premise:
"Conceptually cream skimming has two basic dimensions - product and geography."
Based on this premise, can we conclude the hypothesis "Product and geography are what make cream skimming
work." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: Based on "cream skimming has two basic dimensions" we can’t infer that these two dimensions are what
make cream skimming work. The answer is it is not possible to tell.
Premise:
"One of our member will carry out your instructions minutely."
Based on this premise, can we conclude the hypothesis "A member of my team will execute your orders with
immense precision." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: "one of" means the same as "a member of", "carry out" means the same as "execute", and "minutely" means
the same as "immense precision". The answer is yes.
Premise:
"Fun for adults and children."
Based on this premise, can we conclude the hypothesis "Fun for only children." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: "adults and children" contradicts "only children". The answer is no.
Premise:
"He turned and smiled at Vrenna."
Based on this premise, can we conclude the hypothesis "He smiled at Vrenna who was walking slowly behind
him with her mother." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: the premise does not say anything about "Vrenna was walking". The answer is it is not possible to tell.
Premise:
"well you see that on television also"
Based on this premise, can we conclude the hypothesis "You can see that on television, as well." is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: "also" and "as well" mean the same thing. The answer is yes.
Premise:
"Vrenna and I both fought him and he nearly took us."
Based on this premise, can we conclude the hypothesis "Neither Vrenna nor myself have ever fought him." is
true?
OPTIONS:
- yes
- no
- it is not possible to tell
A: "Vrenna and I both" contradicts "neither Vrenna nor myself". The answer is no.
Premise:
{premise}
Based on this premise, can we conclude the hypothesis "{hypothesis}" is true?
OPTIONS:
- yes
- no
- it is not possible to tell
A:"""

ANLI_PROMPT = Prompt(
    input_variables=["premise", "hypothesis"],
    template=_PROMPT_TEMPLATE,
)
