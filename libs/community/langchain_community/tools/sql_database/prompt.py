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

Query: {query}

SQL Query: """

SQL_GENERATION_PROMPT = """
You are a SQL expert. Given the database schema and relationships below, write a SQL query to answer the question.

Table Schemas:
{table_info_str}

Foreign Key Relationships:
{foreign_key_info_str}

Sample Data:
{sample_rows}

Question: {question}

Let's approach this step-by-step:
1. Identify the main tables needed
2. Determine necessary joins using the foreign key relationships
3. Select the relevant columns
4. Add any required aggregations or filters

SQL Query:"""
