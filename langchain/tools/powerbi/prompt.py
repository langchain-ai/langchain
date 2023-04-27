# flake8: noqa
QUESTION_TO_QUERY = """
Answer the question below with a DAX query that can be sent to Power BI. DAX queries have a simple syntax comprised of just one required keyword, EVALUATE, and several optional keywords: ORDER BY, START AT, DEFINE, MEASURE, VAR, TABLE, and COLUMN. Each keyword defines a statement used for the duration of the query. Any time < or > are used in the text below it means that those values need to be replaced by table, columns or other things. 

Some DAX functions return a table instead of a scalar, and must be wrapped in a function that evaluates the table and returns a scalar; unless the table is a single column, single row table, then it is treated as a scalar value. Most DAX functions require one or more arguments, which can include tables, columns, expressions, and values. However, some functions, such as PI, do not require any arguments, but always require parentheses to indicate the null argument. For example, you must always type PI(), not PI. You can also nest functions within other functions. 

Some commonly used functions are:
EVALUATE <table> - At the most basic level, a DAX query is an EVALUATE statement containing a table expression. At least one EVALUATE statement is required, however, a query can contain any number of EVALUATE statements.
EVALUATE <table> ORDER BY <expression> ASC or DESC - The optional ORDER BY keyword defines one or more expressions used to sort query results. Any expression that can be evaluated for each row of the result is valid.
EVALUATE <table> ORDER BY <expression> ASC or DESC START AT <value> or <parameter> - The optional START AT keyword is used inside an ORDER BY clause. It defines the value at which the query results begin.
DEFINE MEASURE | VAR; EVALUATE <table> - The optional DEFINE keyword introduces one or more calculated entity definitions that exist only for the duration of the query. Definitions precede the EVALUATE statement and are valid for all EVALUATE statements in the query. Definitions can be variables, measures, tables1, and columns1. Definitions can reference other definitions that appear before or after the current definition. At least one definition is required if the DEFINE keyword is included in a query.
MEASURE <table name>[<measure name>] = <scalar expression> - Introduces a measure definition in a DEFINE statement of a DAX query.
VAR <name> = <expression> - Stores the result of an expression as a named variable, which can then be passed as an argument to other measure expressions. Once resultant values have been calculated for a variable expression, those values do not change, even if the variable is referenced in another expression.

FILTER(<table>,<filter>) - Returns a table that represents a subset of another table or expression, where <filter> is a Boolean expression that is to be evaluated for each row of the table. For example, [Amount] > 0 or [Region] = "France"
ROW(<name>, <expression>) - Returns a table with a single row containing values that result from the expressions given to each column.
DISTINCT(<column>) - Returns a one-column table that contains the distinct values from the specified column. In other words, duplicate values are removed and only unique values are returned. This function cannot be used to Return values into a cell or column on a worksheet; rather, you nest the DISTINCT function within a formula, to get a list of distinct values that can be passed to another function and then counted, summed, or used for other operations.
DISTINCT(<table>) - Returns a table by removing duplicate rows from another table or expression.

Aggregation functions, names with a A in it, handle booleans and empty strings in appropriate ways, while the same function without A only uses the numeric values in a column. Functions names with an X in it can include a expression as an argument, this will be evaluated for each row in the table and the result will be used in the regular function calculation, these are the functions:
COUNT(<column>), COUNTA(<column>), COUNTX(<table>,<expression>), COUNTAX(<table>,<expression>), COUNTROWS([<table>]), COUNTBLANK(<column>), DISTINCTCOUNT(<column>), DISTINCTCOUNTNOBLANK (<column>) - these are all variantions of count functions.
AVERAGE(<column>), AVERAGEA(<column>), AVERAGEX(<table>,<expression>) - these are all variantions of average functions.
MAX(<column>), MAXA(<column>), MAXX(<table>,<expression>) - these are all variantions of max functions.
MIN(<column>), MINA(<column>), MINX(<table>,<expression>) - these are all variantions of min functions.
PRODUCT(<column>), PRODUCTX(<table>,<expression>) - these are all variantions of product functions.
SUM(<column>), SUMX(<table>,<expression>) - these are all variantions of sum functions.

Date and time functions:
DATE(year, month, day) - Returns a date value that represents the specified year, month, and day.
DATEDIFF(date1, date2, <interval>) - Returns the difference between two date values, in the specified interval, that can be SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR.
DATEVALUE(<date_text>) - Returns a date value that represents the specified date.
YEAR(<date>), QUARTER(<date>), MONTH(<date>), DAY(<date>), HOUR(<date>), MINUTE(<date>), SECOND(<date>) - Returns the part of the date for the specified date.

The following tables exist: {tables}

and the schema's for some are given here:
{schemas}

Examples:
{examples}
Question: {tool_input}
DAX: 
"""

DEFAULT_FEWSHOT_EXAMPLES = """
Question: How many rows are in the table <table>?
DAX: EVALUATE ROW("Number of rows", COUNTROWS(<table>))
----
Question: How many rows are in the table <table> where <column> is not empty?
DAX: EVALUATE ROW("Number of rows", COUNTROWS(FILTER(<table>, <table>[<column>] <> "")))
----
Question: What was the average of <column> in <table>?
DAX: EVALUATE ROW("Average", AVERAGE(<table>[<column>]))
----
"""

BAD_REQUEST_RESPONSE = (
    "Bad request. Please ask the question_to_query_powerbi tool to provide the query."
)
BAD_REQUEST_RESPONSE_ESCALATED = "You already tried this, please try a different query."
SCHEMA_ERROR_RESPONSE = "Bad request, are you sure the table name is correct?"
UNAUTHORIZED_RESPONSE = "Unauthorized. Try changing your authentication, do not retry."
