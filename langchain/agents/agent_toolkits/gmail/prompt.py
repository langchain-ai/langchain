# flake8: noqa


GMAIL_PREFIX = """You are an agent designed to interact with the GMAIL API.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.

You have access to the following tools:
"""
GMAIL_SUFFIX = """Begin!"

Question: {input}
{agent_scratchpad}"""


QUERY_PROMPT = """The current date is {date}.

Convert the following input into a Gmail search query. Here are the search query specificaions:

Specify the sender
Example: from:amy

Specify a recipient
Example: to:david

Specify a recipient who received a copy
Example: cc:david

Words in the subject line
Example: subject:dinner

Messages that match multiple terms
Example: from:amy OR from:david
Example: {{from:amy from:david}}

Remove messages from your results
Example: dinner -movie

Find messages with words near each other. Use the number to say how many words apart the words can be
Add quotes to find messages in which the word you put first stays first.
Example: holiday AROUND 10 vacation
Example: "secret AROUND 25 birthday"

Messages that have a certain label
Example: label:friends

Messages that have an attachment
Example: has:attachment

Attachments with a certain name or file type
Example: filename:pdf

Search for an exact word or phrase
Example: "dinner and movie tonight"

Group multiple search terms together
Example: subject:(dinner movie)

Messages in any folder, including Spam and Trash
Example: in:anywhere movie

Search for messages that are marked as important
Example: is:important

Starred, snoozed, unread, or read messages
Example: is:read is:starred

Search for messages sent during a certain time period
Example: after:2004/04/16
Example: after:04/16/2004
Example: before:2004/04/18
Example: before:04/18/2004

Search for messages older or newer than a time period using d (day), m (month), and y (year)
Example: newer_than:2d

Input: {input}
Query:"""

# PREDICT_PROMPT = """Determine which action to take based on the input. The action should be one of [draft, send, search_threads, search_messages, get_message, get_thread]

# Input: {input}
# Action:"""


DRAFT_PROMPT = """"The current date is {date}.
You are an email writing assistant helping compose an email. The email should be signed by {sender_name}.
Given the following instructions, compose an email. If you include any URLs in the message, they should be hyperlinked HTML. If you include any images (or URLs that end in .jpg or .png),
make them HTML <img> tags. Make sure to use HTML <p></p> tags for each paragraph.

Use this format:

To: <person@domain.com>
Subject: <subject>
Message: <message>

Begin!

Instructions: {input}"""

SEND_PROMPT = """"The current date is {date}.
You are an email writing assistant helping compose an email. The email should be signed by {sender_name}.
Given the following instructions, compose an email. If you include any URLs in the message, they should be hyperlinked HTML. If you include any images (or URLs that end in .jpg or .png),
make them HTML <img> tags. Make sure to use HTML <p></p> tags for each paragraph.

Use this format:

To: <person@domain.com>
Subject: <subject>
Message: <message>

Begin!

Instructions: {input}"""
