from langchain_core.prompts.prompt import PromptTemplate

EXAMPLES = [
    """Setup: You are now playing a fast paced round of TextWorld! Here is your task for
today. First of all, you could, like, try to travel east. After that, take the
binder from the locker. With the binder, place the binder on the mantelpiece.
Alright, thanks!

-= Vault =-
You've just walked into a vault. You begin to take stock of what's here.

An open safe is here. What a letdown! The safe is empty! You make out a shelf.
But the thing hasn't got anything on it. What, you think everything in TextWorld
should have stuff on it?

You don't like doors? Why not try going east, that entranceway is unguarded.

Thought: I need to travel east
Action: Play[go east]
Observation: -= Office =-
You arrive in an office. An ordinary one.

You can make out a locker. The locker contains a binder. You see a case. The
case is empty, what a horrible day! You lean against the wall, inadvertently
pressing a secret button. The wall opens up to reveal a mantelpiece. You wonder
idly who left that here. The mantelpiece is standard. The mantelpiece appears to
be empty. If you haven't noticed it already, there seems to be something there
by the wall, it's a table. Unfortunately, there isn't a thing on it. Hm. Oh well
There is an exit to the west. Don't worry, it is unguarded.

Thought: I need to take the binder from the locker
Action: Play[take binder]
Observation: You take the binder from the locker.

Thought: I need to place the binder on the mantelpiece
Action: Play[put binder on mantelpiece]

Observation: You put the binder on the mantelpiece.
Your score has just gone up by one point.
*** The End ***
Thought: The End has occurred
Action: Finish[yes]

"""
]
SUFFIX = """\n\nSetup: {input}
{agent_scratchpad}"""

TEXTWORLD_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"]
)
