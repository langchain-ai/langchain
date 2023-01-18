# Google Calendar Wrapper

This page covers how to use the Google Calendar API within LangChain.
It is broken into two parts: installation and setup, and then references to specific Pinecone wrappers.

## Installation and Setup
- Install requirements with `pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`

- Follow instructions here https://developers.google.com/calendar/api/quickstart/python

- Download the `credentials.json` file and save it in the root of your project

## Wrappers

### Utility

There exists a GoogleCalendarAPIWrapper utility which wraps this API. To import this utility:

```python
from langchain.utilities.google_calendar import GoogleCalendarAPIWrapper
```

For a more detailed walkthrough of this wrapper, see [this notebook](../modules/utils/examples/google_calendar.ipynb).

### Tool

You can use it like:

```python
from langchain.utilities.google_calendar import GoogleCalendarAPIWrapper

google_calendar = GoogleCalendarAPIWrapper()

Tool(
    name="Google Calendar",
    func=google_calendar.run,
    description="Useful for when you need to perform an action in a calendar. The input should be the initial query you want to ask the calendar.",
),
```

