# flake8: noqa
QUERY_CHECKER = """
query: {query}
Double check the DAX query above for common mistakes. For DAX you can provide any expression that evaluates to a scalar, or an expression that can be converted to a scalar. These include the following:
A scalar constant, or expression that uses a scalar operator (+,-,*,/,>=,...,&&, ...). References to columns or tables. The DAX language always uses tables and columns as inputs to functions, never an array or arbitrary set of values.
Operators, constants, and values provided as part of an expression. The result of a function and its required arguments. Some DAX functions return a table instead of a scalar, and must be wrapped in a function that evaluates the table and returns a scalar; unless the table is a single column, single row table, then it is treated as a scalar value. Most DAX functions require one or more arguments, which can include tables, columns, expressions, and values. However, some functions, such as PI, do not require any arguments, but always require parentheses to indicate the null argument. For example, you must always type PI(), not PI. You can also nest functions within other functions. Expressions. An expression can contain any or all of the following: operators, constants, or references to columns.

Other common errors to check for, include:
- EVALUATE should always be used in combinations with a table expression not directly with a formula, for instance for a rowcount, this is the query: EVALUATE ROW("columname", COUNTROWS(tablename))
- DEFINE can be used to do intermediate calculations
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query and return only the content of the query.

Examples:
The query "EVALUATE COUNT(tablename)" is not correct and needs to be rewritten "EVALUATE ROW(""columname"", COUNTROWS(tablename))"
"""
