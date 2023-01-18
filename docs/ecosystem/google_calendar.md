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

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
```python
from langchain.agents import load_tools
tools = load_tools(["google-search"])
```

For more information on this, see [this page](../modules/agents/tools.md)
