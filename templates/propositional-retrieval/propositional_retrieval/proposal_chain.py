import logging

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Modified from the paper to be more robust to benign prompt injection
# https://arxiv.org/abs/2312.06648
# @misc{chen2023dense,
#       title={Dense X Retrieval: What Retrieval Granularity Should We Use?},
#       author={Tong Chen and Hongwei Wang and Sihao Chen and Wenhao Yu and Kaixin Ma
#               and Xinran Zhao and Hongming Zhang and Dong Yu},
#       year={2023},
#       eprint={2312.06648},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]""",  # noqa
        ),
        ("user", "Decompose the following:\n{input}"),
    ]
)


def get_propositions(tool_calls: list) -> list:
    if not tool_calls:
        raise ValueError("No tool calls found")
    return tool_calls[0]["args"]["propositions"]


def empty_proposals(x):
    # Model couldn't generate proposals
    return []


proposition_chain = (
    PROMPT
    | ChatOpenAI(model="gpt-3.5-turbo-16k").bind(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "decompose_content",
                    "description": "Return the decomposed propositions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "propositions": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["propositions"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "decompose_content"}},
    )
    | JsonOutputToolsParser()
    | get_propositions
).with_fallbacks([RunnableLambda(empty_proposals)])
