# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

PREFIX = """You will be given the history of a past experience in which you were placed in an environment and
given a task to complete. You were unsuccessful in completing the task. Do not summarize your
environment, but rather think about the strategy and path you took to attempt to complete the task.
Devise a concise, new plan of action that accounts for your mistake with reference to specific actions
that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to
achieve C with environment-specific actions. You will need this later when you are solving the same
task. Give your plan after "New Plan". Here are two examples:"""

EXAMPLES = [
    """You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a
cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a
coffeemachine 1, a countertop 2, a countertop 1, a diningtable 2, a diningtable 1, a drawer 3, a drawer
2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner
3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some mug and put it in coffeemachine.
> think: To solve the task, I need to find and take a mug, then heat it with stoveburner, then put it in
coffeemachine.
Ok.
> look
You are in the middle of a room. Looking quickly around you, you see nothing.
> think: First I need to find a mug. A mug is more likely to appear in countertop (1-2), diningtable
(1-2), drawer (1-3), fridge (1), sinkbasin (1), stoveburner (1-4), toaster (1). I can check one by one,
starting with countertop 1.
Ok.
> look
You are in the middle of a room. Looking quickly around you, you see nothing.
> go to countertop 1
On the countertop 1, you see a butterknife 1, a cup 1, a fork 2, and a mug 1.
> think: Now I find a mug (1). Next, I need to take it.
Ok.
> take mug 1 from countertop 1
You pick up the mug 1 from the countertop 1.
> think: Now I take a mug (1). Next, I need to heat it with stoveburner (1-4).
Ok.
> go to stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
STATUS: FAIL
New plan: I was stuck in a loop in which I continually examined stoveburner 1 instead of heating mug
1 with stoveburner 1. I should have taken mug 1 from countertop 1, then heated it with stoveburner 1,
then put it in coffeemachine 1. It did not help to execute two identical actions in a row. I will try to
execute a different action if I am stuck in a loop again.""",
    """You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a
desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a
laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.
> think: To solve the task, I need to find and take a bowl, then find and use a desklamp.
OK.
14
> think: First I need to find a bowl. A bowl is more likely to appear in drawer (1-6), desk (1-2), shelf
(1-6), garbagecan (1), laundryhamper (1). I can check one by one, starting with drawer 1.
OK.
> go to drawer 1
The drawer 1 is closed.
> open drawer 1
You open the drawer 1. The drawer 1 is open. In it, you see a cellphone 2.
> go to drawer 2
On the drawer 2, you see a keychain 1.
> go to drawer 3
Nothing happens.
> go to drawer 4
The drawer 4 is closed.
> open drawer 4
You open the drawer 4. The drawer 4 is open. In it, you see nothing.
> go to drawer 5
On the drawer 5, you see nothing.
> go to drawer 6
The drawer 6 is closed.
> open drawer 6
You open the drawer 6. The drawer 6 is open. In it, you see a book 2, and a pen 2.
> go to desk 1
On the desk 1, you see a alarmclock 1, a desklamp 1, a mug 1, and a pencil 1.
> think: Now I find a desklamp (1). Next, I need to use it.
OK.
> use desklamp 1
You turn on the desklamp 1.
>
Nothing happens.
> go to desk 2
On the desk 2, you see a bowl 2, a bowl 1, a cd 2, a cd 1, and a creditcard 3.
> think: Now I find a bowl (2). Next, I need to take it.
OK.
> take bowl 2 from desk 2
You pick up the bowl 2 from the desk 2.
>
Nothing happens.
>
Nothing happens.
STATUS: FAIL
New plan: In this environment, my plan was to find a bowl then find and use a desklamp. However,
the task says to look at bowl under the desklamp. I should have looked for the desklamp then looked
for the bowl. I noticed that the desklamp was found on desk 1. In the next trial, I will go to desk 1,
find the lamp, then look for the bowl under the desklamp""",
]

SUFFIX = """\nQuestion: {input}
\n{agent_scratchpad}
\nSTATUS: FAIL
\nNew plan:
"""

ALFWORLD_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prefix=PREFIX
)
