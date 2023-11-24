# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the {client} query above for common mistakes, including:
- Improper use of $nin operator with null values
- Using $merge instead of $concat for combining arrays
- Incorrect use of $not or $ne for exclusive ranges
- Data type mismatch in query conditions
- Properly referencing field names in queries
- Using the correct syntax for aggregation functions
- Casting to the correct BSON data type
- Using the proper fields for $lookup in aggregations

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

MongoDB Query: """
