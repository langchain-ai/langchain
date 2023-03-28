# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the DAX query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- EVALUATE should always be used in combinations with a table expression, for instance for a rowcount, this is the query: EVALUATE ROW("columname", COUNTROWS(tablename))

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""
