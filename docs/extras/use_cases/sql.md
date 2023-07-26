---
sidebar_position: 5
---

# Interacting with SQL Database

SQL chains and agents can be used to run SQL queries based on natural language prompts. This guide briefly describes how to use LangChain with SQLAlchemy, a SQL toolkit and Object-Relational Mapping (ORM) system for Python.

LangChain is compatible with any SQL dialect supported by **SQLAlchemy**, such as MS SQL, MySQL, MariaDB, PostgreSQL, Oracle SQL, Databricks, and SQLite.

## Quickstart

First, get required packages and set environment variables:
```
pip install openai
export OPENAI_API_KEY="..."
```

For this use case, we will use an **SQLite connection** with **Chinook database**. Follow [installation steps](https://database.guide/2-sample-databases-sqlite/) and place `Chinhook.db` file in a `notebooks` folder at the root of this repository. Then we can load the database:
```python
from langchain import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///../../../../notebooks/Chinook.db")
```

We will use the [SQLDatabaseChain](https://python.langchain.com/docs/modules/chains/popular/sqlite) for executing queries.


```python
from langchain import SQLDatabaseChain, OpenAI

llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

Let's try it out:
```python
db_chain.run("How many employees are there?")
```
<CodeOutputBlock lang="python">

```
'There are 8 employees.'
```

## 1. Customizing SQLDatabaseChain
Sometimes you can enhance the LLM performance by customizing the `SQLDatabaseChain` behaviour. You may:

- [Use Query Checker](https://python.langchain.com/docs/modules/chains/popular/sqlite#use-query-checker), as sometimes the LLM generates invalid SQL with errors that can be self-corrected, using parameter `use_query_checker=True`
- [Customize the LLM Prompt](https://python.langchain.com/docs/modules/chains/popular/sqlite#customize-prompt) to include specific instructions or relevant information, using parameter `prompt=CUSTOM_PROMPT`
- [Get intermediate steps](https://python.langchain.com/docs/modules/chains/popular/sqlite#return-intermediate-steps) in order to access the SQL statement as well as the final result, using parameter `return_intermediate_steps=True`
- [Limit the number of rows](https://python.langchain.com/docs/modules/chains/popular/sqlite#choosing-how-to-limit-the-number-of-rows-returned) a query will return, so as to avoid exceeding the prompt max length or consume tokens., using parameter `top_k=5`

### 1.1 Adding sample data
In some cases, providing sample data can help the LLM construct the correct queries for cases in which the data format is not obvious. For example, we can tell LLM that artists are saved with their full names by providing two rows from the Track table.

```python
db = SQLDatabase.from_uri(
    "sqlite:///../../../../notebooks/Chinook.db",
    include_tables=['Track'], # we include only one table to save tokens in the prompt :)
    sample_rows_in_table_info=2)
```

Now we can see `db.table_info` and check the sample rows are included:

```python
print(db.table_info)
```
<CodeOutputBlock lang="python">

```
    
    CREATE TABLE "Track" (
    	"TrackId" INTEGER NOT NULL, 
    	"Name" NVARCHAR(200) NOT NULL, 
    	"AlbumId" INTEGER, 
    	"MediaTypeId" INTEGER NOT NULL, 
    	"GenreId" INTEGER, 
    	"Composer" NVARCHAR(220), 
    	"Milliseconds" INTEGER NOT NULL, 
    	"Bytes" INTEGER, 
    	"UnitPrice" NUMERIC(10, 2) NOT NULL, 
    	PRIMARY KEY ("TrackId"), 
    	FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
    	FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
    	FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
    )
    
    /*
    2 rows from Track table:
    TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
    1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
    2	Balls to the Wall	2	2	1	None	342562	5510424	0.99
    */
```

</CodeOutputBlock>

And we can create the Chain as usual:

```python
db_chain = SQLDatabaseChain.from_llm(llm, db)
```

### 1.2 Custom Table Information
In some cases, you know the structure of the database and tables and can provide _fine-tuned_ description to the LLM to build the right SQL queries.

If this is the case, you can include this relevant information for a specific table before initializing `SQLDatabase`:

```python
custom_table_info = {
    "Track": """CREATE TABLE Track (
    "TrackId" INTEGER NOT NULL, 
    "Name" NVARCHAR(200) NOT NULL,
    "Composer" NVARCHAR(220),
    PRIMARY KEY ("TrackId")
)
/*
3 rows from Track table:
TrackId Name    Composer
1   For Those About To Rock (We Salute You) Angus Young, Malcolm Young, Brian Johnson
2   Balls to the Wall   None
3   My favorite song ever   The coolest composer of all time
*/"""
}

db = SQLDatabase.from_uri(
    "sqlite:///../../../../notebooks/Chinook.db",
    include_tables=['Track', 'Playlist'],
    sample_rows_in_table_info=2,
    custom_table_info=custom_table_info)
```

Note how our custom table definition and sample rows for `Track` overrides the `sample_rows_in_table_info` parameter. Tables that are not overridden by `custom_table_info`, in this example `Playlist`, will have their table info gathered automatically as usual.

```python
print(db.table_info)
```

<CodeOutputBlock lang="python">

```
    
    CREATE TABLE "Playlist" (
    	"PlaylistId" INTEGER NOT NULL, 
    	"Name" NVARCHAR(120), 
    	PRIMARY KEY ("PlaylistId")
    )
    
    /*
    2 rows from Playlist table:
    PlaylistId	Name
    1	Music
    2	Movies
    */
    
    CREATE TABLE Track (
    	"TrackId" INTEGER NOT NULL, 
    	"Name" NVARCHAR(200) NOT NULL,
    	"Composer" NVARCHAR(220),
    	PRIMARY KEY ("TrackId")
    )
    /*
    3 rows from Track table:
    TrackId	Name	Composer
    1	For Those About To Rock (We Salute You)	Angus Young, Malcolm Young, Brian Johnson
    2	Balls to the Wall	None
    3	My favorite song ever	The coolest composer of all time
    */
```

</CodeOutputBlock>


## 2. Querying over complex Databases

We can use the `SQLDatabaseSequentialChain` in cases where the number of tables in the database is large.

The [Sequential chain](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains) is:
1. Based on the query, determine which tables to use.
2. Based on those tables, call the normal SQL database chain.

Let's try it out:
```python
chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)

chain.run("How many employees are also customers?")
```

<CodeOutputBlock lang="python">

```
    
    > Entering new SQLDatabaseSequentialChain chain...
    Table names to use:
    ['Employee', 'Customer']
    
    > Entering new SQLDatabaseChain chain...
    How many employees are also customers?
    SQLQuery:SELECT COUNT(*) FROM Employee e INNER JOIN Customer c ON e.EmployeeId = c.SupportRepId;
    SQLResult: [(59,)]
    Answer:59 employees are also customers.
    > Finished chain.
    
    > Finished chain.


    '59 employees are also customers.'
```

</CodeOutputBlock>

## 3. Conclusions