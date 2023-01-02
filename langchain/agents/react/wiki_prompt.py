# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

EXAMPLES = [
    """Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]""",
    """Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons"
character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after
who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated
television series The Simpsons voiced by Pamela Hayden and created by Matt
Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up
"named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose
middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is
Richard Nixon.
Action 3: Finish[Richard Nixon]""",
    """Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The
Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which
documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1 Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell
III’, ’Seventh Avenue (Manhattan)’, ’Adam Clayton Powell Jr. State Office
Building’, ’Isabel Washington Powell’, ’Adam Powell’, ’Adam Clayton Powell
(film)’, ’Giancarlo Esposito’].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by
Richard Kilberg. The film is about the rise and fall of influential
African-American politician Adam Clayton Powell Jr.[3][4] It was later aired
as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American
politician, not Finnish rock groups. So the documentary about Finnish rock
groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]""",
    """Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then
find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16,
1979) was an American film director, screenwriter, and actor best known for
the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need
to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter
and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor.
So profession Nicholas Ray and Elia Kazan have in common is director,
screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]""",
    """Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought 1: I need to search Arthur’s Magazine and First for Women, and find which was
started first.
Action 1: Search[Arthur’s Magazine]
Observation 1: Arthur’s Magazine (1844-1846) was an American literary periodical published
in Philadelphia in the 19th century.
Thought 2: Arthur’s Magazine was started in 1844. I need to search First for Women
next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman’s magazine published by Bauer Media Group in the
USA.[1] The magazine was started in 1989.
Thought 3: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First
for Women), so Arthur’s Magazine was started first.
Action 3: Finish[Arthur’s Magazine]""",
    """Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work,
then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet
mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and
find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer
scientist.
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn
and Leonid Levin have the same type of work.
Action 3: Finish[yes]""",
]
SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""

WIKI_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"]
)
