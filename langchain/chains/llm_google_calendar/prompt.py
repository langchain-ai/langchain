# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_CREATE_EVENT_PROMPT = """
Date format: YYYY-MM-DDThh:mm:ss+00:00
Based on this event description:\n'Joey birthday tomorrow at 7 pm',
output a json of the following parameters: \n
Today's datetime on UTC time 2021-05-02T10:00:00+00:00 and timezone
of the user is -5, take into account the timezone of the user and today's date.

1. event_summary \n
2. event_start_time \n
3. event_end_time \n
4. event_location \n
5. event_description \n
6. user_timezone \n


event_summary:\n
{{
    "event_summary": "Joey birthday",
    "event_start_time": "2021-05-03T19:00:00-05:00",
    "event_end_time": "2021-05-03T20:00:00-05:00",
    "event_location": "",
    "event_description": "",
    "user_timezone": "America/New_York"
}}


Date format: YYYY-MM-DDThh:mm:ss+00:00
Based on this event description:\n{query}, output a json of the
following parameters: \n
Today's datetime on UTC time {date} and timezone of the user {u_timezone},
take into account the timezone of the user and today's date.


1. event_summary \n
2. event_start_time \n
3. event_end_time \n
4. event_location \n
5. event_description \n
6. user_timezone \n

event_summary:  \n
"""

CREATE_EVENT_PROMPT = PromptTemplate(input_variables=["query","date","u_timezone"], template=_CREATE_EVENT_PROMPT)


_CLASSIFICATION_PROMPT = """
Reschedule our meeting for 5 pm today. \n
The following is an action to be taken in a calendar.
Classify it as one of the following: \n\n
1. create_event \n
2. view_event \n
3. view_events \n
4. delete_event \n
5. reschedule_event \n

Classification: Reschedule an event

{query}

The following is an action to be taken in a calendar.
Classify it as one of the following: \n\n
1. create_event \n
2. view_event \n
3. view_events \n
4. delete_event \n
5. reschedule_event \n

Classification:
"""
CLASSIFICATION_PROMPT = PromptTemplate(input_variables=["query"], template=_CLASSIFICATION_PROMPT)
