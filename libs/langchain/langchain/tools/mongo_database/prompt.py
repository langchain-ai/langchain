# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the MongoDB query above for common mistakes, including:
- Not a valid Python dictionary/MongoDB document
- Not an existing MongoDB command
- Improper use of $nin operator with null values
- Using $merge instead of $concat for combining arrays
- Incorrect use of $not or $ne for exclusive ranges
- Data type mismatch in query conditions
- Improperly referencing field names in queries
- Using incorrect syntax for aggregation functions
- Casting to the incorrect BSON data type
- Using the improper fields for $lookup in aggregations

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

MongoDB Query: """
