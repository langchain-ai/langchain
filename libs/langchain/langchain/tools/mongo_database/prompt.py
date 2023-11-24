# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the MongoDB query above for common mistakes, including:
- Correct syntax for query operators (e.g., $match, $group, $project)
- Properly matching nested fields in the documents
- Using the appropriate array operators (e.g., $elemMatch)
- Utilizing indexes for performance optimization
- Handling data type mismatch in queries
- Ensuring proper field names and key names in queries
- Using the correct projection operators for desired output
- Properly structuring aggregation pipelines if applicable

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

MongoDB Query: """