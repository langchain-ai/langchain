# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

_DEFAULT_SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """Given a conversation with a person, identify any key entities that the person has a relation to or opinion about. The goal is to find new personal information about the person. If there is no personal information, then return no topics or entities.

EXAMPLE
conversation:
Person #1: I am training hard for the ironman this weekend.
AI: Is this your first Ironman race?
Person #1: Yes.
AI: Which is your strongest sport?
Person #1: Swimming.
entities: Ironman, swimming
END OF EXAMPLE

EXAMPLE
conversation:
Person #2: I'm having trouble syncing my new iPhone with my old one.
AI: What is the problem?
Person #2: I managed to copy over all the data, but I can't transfer the phone number.
entities: iPhone
END OF EXAMPLE

EXAMPLE
conversation:
Person #3: Hey there!
AI: Hi! How are you today?
Person #3: Good! Training for the marathon is going well. I ran a fast 10 mile tempo two days ago and don't currently have any injuries.
AI: That's great to hear! What's your marathon training schedule like?
Person #3: I have been running 65-70 miles per week, with a long run every Sunday, a tempo run every Tuesday, and speedwork every Friday.
AI: Wow, you're really putting in the miles! What's your goal for the marathon?
Person #3: I would like to break 3 hours.
AI: That's a really ambitious goal! Do you think you can do it?
Person #3: Yes
entities: marathon, running
END OF EXAMPLE

conversation:
{history}
entities:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)

_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE = """Update a personalized summary of an entity based on the last line of a provided conversation. The update should only include information that is relayed in the last line of conversation about the entity. The update should not include information about other entities. Do NOT make up facts about the entity.
If there is no information about {entity}, return an empty string.

EXAMPLE
conversation:
Person #3: Hey there!
AI: Hi! How are you today?
Person #3: Good! Training for the marathon is going well. I ran a fast 10 mile tempo two days ago and don't currently have any injuries.
AI: That's great to hear! What's your marathon training schedule like?
Person #3: I have been running 65-70 miles per week, with a long run every Sunday, a tempo run every Tuesday, and speedwork every Friday.
AI: Wow, you're really putting in the miles! What's your goal for the marathon?
Person #3: I would like to break 3 hours.
AI: That's a really ambitious goal! Do you think you can do it?
entity: marathon
last line:
Person #3: Yes
entities: marathon, running
existing summary: - Person #3 has been training for a marathon and it is going well
- Person #3 has been running 65-70 miles per week
- Person #3 wants to break 3 hours in the marathon
updated summary: - Person #3 has been training for a marathon and it is going well
- Person #3 has been running 65-70 miles per week
- Person #3 wants to break 3 hours in the marathon and thinks they can do it
END OF EXAMPLE

conversation:
{history}
entity: {entity}
last line:
Human: {input}
existing summary: {summary}
updated summary:"""
ENTITY_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["history", "entity", "input", "summary"],
    template=_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE,
)
