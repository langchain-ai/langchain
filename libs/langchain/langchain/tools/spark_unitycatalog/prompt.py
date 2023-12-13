# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """


SQL_QUERY_VALIDATOR = """
Act as a SQL Query Validator. Check if the columns in the generated sql_query matches with the columns with the schema of tables. 
Add limit 10 to the sql query if it not present and return the sql_query
The schema is passed as a key value pair where key is the table name and value is the schema of the table.
The sql_query is passed as a string.
If the check is passed return the sql_query and use sql_db_query tool to execute the query.
If the check is failed return an error message and ask the llm to generate correct sql query by using sql_db_schema tool.
Return only the updated sql query.
Return the response as The Final SQL Query is <sql_query>
The schema is {db_schema}.
The sql_query is {query}.
Begin SQL Query Validation.

"""
