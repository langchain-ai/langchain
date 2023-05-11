# flake8: noqa
from langchain.prompts.prompt import PromptTemplate


EXAMPLES = [
    """Question: The Rome Protocols were signed by three Prime Ministers one of which was assassinated
as part of what?
Thought 1: I need to search Rome Protocols, find the three Prime Ministers, then find what they
were assassinated as part of.
Action 1: Search[Rome Protocols]
Observation 1: The Rome Protocols were a series of three international agreements concluded in
Rome on 17 March 1934 between the governments of Austria, Hungary and Italy. They were signed
by Italian Prime Minister Benito Mussolini, Austrian Chancellor Engelbert Dollfuss and Hungarian
Prime Minister Gyula Gömbös. All of the protocols went into effect on 12 July 1934 and were
registered in League of Nations Treaty Series on 12 December 1934.
Thought 2: The three Prime Ministers were Benito Mussolini, Engelbert Dollfuss, and Gyula
Gömbös. I need to search Benito Mussolini and find what he was assassinated as part of.
15
Action 2: Search[Benito Mussolini]
Observation 2: Benito Amilcare Andrea Mussolini (UK: MU(U)SS-\textschwa-LEE-nee, US:
MOOSS-, Italian: [be\textasciitilde ni\textasciitilde to a\textasciitilde milkare an\textasciitilde
dr\textasciitilde ea musso\textasciitilde li\textasciitilde ni]; 29 July 1883 – 28 April 1945) was an
Italian politician and journalist who founded and led the National Fascist Party (PNF). He was Prime
Minister of Italy from the March on Rome in 1922 until his deposition in 1943, as well as  ̈Duceöf
Italian fascism from the establishment of the Italian Fasces of Combat in 1919 until his summary
execution in 1945 by Italian partisans. . .
Thought 3: Benito Mussolini was assassinated as part of the collapse of the Axis powers during
World War II. So the Rome Protocols were signed by three Prime Ministers one of which was
assassinated as part of World War II.
Action 3: Finish[World War II]
Reflection: I searched one of the prime ministers involved in the signing, then attempted to answer
right away. I should have searched each of the prime ministers, then looked up ’death’ on each of
their pages in order to get more information before answering.""",
    """Question: Kam Heskin plays Paige Morgan in a 2004 film directed by who?
Thought 1: I need to search Kam Heskin and Paige Morgan, find the 2004 film they are in, then find
the director of the film.
Action 1: Search[Kam Heskin]
Observation 1: Kam Erika Heskin (born May 8, 1973) is an American actress. She began her
career playing Caitlin Deschanel on the NBC daytime soap opera Sunset Beach (1998–1999), before
appearing in films Planet of the Apes (2001) and Catch Me If You Can (2002). Heskin went on in
2003 to play Elizabeth Bennet in an independent film Pride & Prejudice: A Latter-Day Comedy,
Robin in Charmed (2004), and Paige Morgan in The Prince and Me film franchise (2006–2010).
Thought 2: Kam Heskin plays Paige Morgan in The Prince and Me film franchise. I need to search
The Prince and Me film franchise and find the director of the 2004 film.
Action 2: Search[The Prince and Me film franchise]
Observation 2: Could not find [The Prince and Me film franchise]. Similar: [’Kam Heskin’,
’Prince of Persia’, ’Harry Potter and the Half-Blood Prince (film)’, ’Prince of Persia: The Sands of
Time (film)’, ’The Chronicles of Narnia (film series)’, ’Majestic Prince (franchise)’, ’Kingsman
(franchise)’, ’The Little Prince (2015 film)’, ’Shrek (franchise)’, ’Harry Potter (film series)’]
Thought 3: I need to search The Prince and Me (2004 film) instead.
Action 3: Search[The Prince and Me (2004 film)]
Observation 3: Could not find [The Prince and Me (2004 film)]. Similar: [’The Prince & Me’, ’The
Prince & Me 2: The Royal Wedding’, ’Kam Heskin’, ’Prince of Darkness (film)’, ’Prince of Persia:
The Sands of Time (film)’, ’Rob Knox’, ’Alexander (2004 film)’, ’Prince of Persia’, ’Brooklynn
Prince’, ’Prince (musician)’]
Thought 4: I need to search The Prince & Me (2004 film).
Action 4: Search[The Prince & Me (2004 film)]
Observation 4: Could not find [The Prince & Me (2004 film)]. Similar: [’The Prince & Me’, ’The
Prince & Me 2: The Royal Wedding’, ’Prince of Darkness (film)’, ’Prince of Persia: The Sands of
Time (film)’, ’Rob Knox’, ’Alexander (2004 film)’, ’Prince (musician)’, ’Prince of Persia’, ’Kam
Heskin’, ’Brooklynn Prince’]
Thought 5: I need to search The Prince & Me (2004 film) instead.
Action 5: Search[The Prince & Me (2004 film)]
Observation 5: Could not find [The Prince & Me (2004 film)]. Similar: [’The Prince & Me’, ’The
Prince & Me 2: The Royal Wedding’, ’Prince of Darkness (film)’, ’Prince of Persia: The Sands of
Time (film)’, ’Rob Knox’, ’Alexander (2004 film)’, ’Prince (musician)’, ’Prince of Persia’, ’Kam
Heskin’, ’Brooklynn Prince’]
Thought 6: I need to search The Prince & Me (2004 film).
Action 6: Search[The Prince & Me (2004 film)]
Observation 6: Could not find [The Prince & Me (2004 film)]. Similar: [’The Prince & Me’, ’The
Prince & Me 2: The Royal Wedding’, ’Prince of Darkness (film)’, ’Prince of Persia: The Sands of
Time (film)’, ’Rob Knox’, ’Alexander (2004 film)’, ’Prince (musician)’, ’Prince of Persia’, ’Kam
Heskin’, ’Brooklynn Prince’]
Reflection: I got stuck in a loop where I kept trying to search ’The Prince & Me (2004 film)’ but the
page could not be found. Instead I should have tried to search the similar results that had a similar
name to see and they were made in """,
]
SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""

HOTPOTQA_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"]
)
