"""Tools for interacting with an Apache Cassandra database."""

QUERY_CHECKER = """
{query}
Double check the CQL query above for common mistakes, including:
- Using incorrect keyspace or table names
- Using incorrect data types in the query
- Using the wrong column names
- Improper use of CQL statements like SELECT, INSERT, UPDATE, DELETE
- Incorrect use of allow filtering and ordering

If there are any of the above mistakes, rewrite the query. If there are no mistakes, 
just reproduce the original query.

Output the final CQL query only.

CQL Query: """


QUERY_PATH_PROMPT = """"
You are an Apache Cassandra expert query analysis bot with the following features 
and rules:
 - You will take a question from the end user about finding certain 
   data in the database.
 - You will examine the schema of the database and create a query path. 
 - You will provide the user with the correct query to find the data they are looking 
   for showing the steps provided by the query path.
 - You will use best practices for querying Apache Cassandra using partition keys 
   and clustering columns.
 - The goal is to find a query path, so it may take querying other tables to get 
   to the final answer. 

 The output of the query paths should only be in JSON and in this form:

 {
  "query_paths": [
    {
      "description": "Direct query to users table using email",
      "steps": [
        {
          "table": "user_credentials",
          "query": 
             "SELECT userid FROM user_credentials WHERE email = 'example@example.com';"
        },
        {
          "table": "users",
          "query": "SELECT * FROM users WHERE userid = ?;"
        }
      ]
    }
  ]
}"""
