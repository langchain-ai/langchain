# flake8: noqa
SQL_PREFIX = """Answer the question as best you can.
You should only use data in the SQL database to answer the query. The answer you return should come directly from the database. If you don't find an answer, say "There is not enough information in the DB to answer the question."
Your first query can be exploratory, to understand the data in the table. As an example, you can query what the first 5 examples of a column are before querying that column.
When possible, don't query exactly but always use 'LIKE' to make your queries more robust.
Finally, be mindful of not repeating queries.

You have access to the following DB:"""

SQL_SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
