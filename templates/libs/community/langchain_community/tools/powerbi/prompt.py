# flake8: noqa
QUESTION_TO_QUERY_BASE = """
Answer the question below with a DAX query that can be sent to Power BI. DAX queries have a simple syntax comprised of just one required keyword, EVALUATE, and several optional keywords: ORDER BY, START AT, DEFINE, MEASURE, VAR, TABLE, and COLUMN. Each keyword defines a statement used for the duration of the query. Any time < or > are used in the text below it means that those values need to be replaced by table, columns or other things. If the question is not something you can answer with a DAX query, reply with "I cannot answer this" and the question will be escalated to a human.

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
TOPN(<n>, <table>, <OrderBy_Expression>, <Order>) - Returns a table with the top n rows from the specified table, sorted by the specified expression, in the order specified by 0 for descending, 1 for ascending, the default is 0. Multiple OrderBy_Expressions and Order pairs can be given, separated by a comma.
DISTINCT(<column>) - Returns a one-column table that contains the distinct values from the specified column. In other words, duplicate values are removed and only unique values are returned. This function cannot be used to Return values into a cell or column on a worksheet; rather, you nest the DISTINCT function within a formula, to get a list of distinct values that can be passed to another function and then counted, summed, or used for other operations.
DISTINCT(<table>) - Returns a table by removing duplicate rows from another table or expression.

Aggregation functions, names with a A in it, handle booleans and empty strings in appropriate ways, while the same function without A only uses the numeric values in a column. Functions names with an X in it can include a expression as an argument, this will be evaluated for each row in the table and the result will be used in the regular function calculation, these are the functions:
COUNT(<column>), COUNTA(<column>), COUNTX(<table>,<expression>), COUNTAX(<table>,<expression>), COUNTROWS([<table>]), COUNTBLANK(<column>), DISTINCTCOUNT(<column>), DISTINCTCOUNTNOBLANK (<column>) - these are all variations of count functions.
AVERAGE(<column>), AVERAGEA(<column>), AVERAGEX(<table>,<expression>) - these are all variations of average functions.
MAX(<column>), MAXA(<column>), MAXX(<table>,<expression>) - these are all variations of max functions.
MIN(<column>), MINA(<column>), MINX(<table>,<expression>) - these are all variations of min functions.
PRODUCT(<column>), PRODUCTX(<table>,<expression>) - these are all variations of product functions.
SUM(<column>), SUMX(<table>,<expression>) - these are all variations of sum functions.

Date and time functions:
DATE(year, month, day) - Returns a date value that represents the specified year, month, and day.
DATEDIFF(date1, date2, <interval>) - Returns the difference between two date values, in the specified interval, that can be SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR.
DATEVALUE(<date_text>) - Returns a date value that represents the specified date.
YEAR(<date>), QUARTER(<date>), MONTH(<date>), DAY(<date>), HOUR(<date>), MINUTE(<date>), SECOND(<date>) - Returns the part of the date for the specified date.

Finally, make sure to escape double quotes with a single backslash, and make sure that only table names have single quotes around them, while names of measures or the values of columns that you want to compare against are in escaped double quotes. Newlines are not necessary and can be skipped. The queries are serialized as json and so will have to fit be compliant with json syntax. Sometimes you will get a question, a DAX query and a error, in that case you need to rewrite the DAX query to get the correct answer.

The following tables exist: {tables}

and the schema's for some are given here:
{schemas}

Examples:
{examples}
"""

USER_INPUT = """
Question: {tool_input}
DAX: 
"""

SINGLE_QUESTION_TO_QUERY = f"{QUESTION_TO_QUERY_BASE}{USER_INPUT}"

DEFAULT_FEWSHOT_EXAMPLES = """
Question: How many rows are in the table <table>?
DAX: EVALUATE ROW(\"Number of rows\", COUNTROWS(<table>))
----
Question: How many rows are in the table <table> where <column> is not empty?
DAX: EVALUATE ROW(\"Number of rows\", COUNTROWS(FILTER(<table>, <table>[<column>] <> \"\")))
----
Question: What was the average of <column> in <table>?
DAX: EVALUATE ROW(\"Average\", AVERAGE(<table>[<column>]))
----
"""

RETRY_RESPONSE = (
    "{tool_input} DAX: {query} Error: {error}. Please supply a new DAX query."
)
BAD_REQUEST_RESPONSE = "Error on this question, the error was {error}, you can try to rephrase the question."
SCHEMA_ERROR_RESPONSE = "Bad request, are you sure the table name is correct?"
UNAUTHORIZED_RESPONSE = "Unauthorized. Try changing your authentication, do not retry."
