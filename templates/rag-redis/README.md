# Redis RAG Example

Using Langserve and Redis to build a RAG search example for answering questions on financial 10k filings docs (for Nike).

Relies on the sentence transformer `all-MiniLM-L6-v2` for embedding chunks of the pdf and user questions.

## Running Redis

There are a number of ways to run Redis depending on your use case and scenario.

### Easiest? Redis Cloud

Create a free database on [Redis Cloud](https://redis.com/try-free). *No credit card information is required*. Simply fill out the info form and select the cloud vendor of your choice and region.

Once you have created an account and database, you can find the connection credentials by clicking on the database and finding the "Connect" button which will provide a few options. Below are the environment variables you need to configure to run this RAG app.

```bash
export REDIS_HOST = <YOUR REDIS HOST>
export REDIS_PORT = <YOUR REDIS PORT>
export REDIS_USER = <YOUR REDIS USER NAME>
export REDIS_PASSWORD = <YOUR REDIS PASSWORD>
```

For larger use cases (greater than 30mb of data), you can certainly created a Fixed or Flexible billing subscription which can scale with your dataset size.

### Redis Stack -- Local Docker

For local development, you can use Docker:

```bash
docker run -p 6397:6397 -p 8001:8001 redis/redis-stack:latest
```

This will run Redis on port 6379. You can then check that it is running by visiting the RedisInsight GUI at [http://localhost:8001](http://localhost:8001).

This is the connection that the application will try to use by default -- local dockerized Redis.

## Data

To load the financial 10k pdf (for Nike) into the vectorstore, run the following command from the root of this repository:

```bash
poetry shell
python ingest.py
```

## Supported Settings
We use a variety of environment variables to configure this application

| Environment Variable | Description                       | Default Value |
|----------------------|-----------------------------------|---------------|
| `DEBUG`            | Enable or disable Langchain debugging logs       | True         |
| `REDIS_HOST`           | Hostname for the Redis server     | "localhost"   |
| `REDIS_PORT`           | Port for the Redis server         | 6379          |
| `REDIS_USER`           | User for the Redis server         | "" |
| `REDIS_PASSWORD`       | Password for the Redis server     | "" |
| `REDIS_URL`            | Full URL for connecting to Redis  | `None`, Constructed from user, password, host, and port if not provided |
| `INDEX_NAME`           | Name of the vector index          | "rag-redis"   |



## Installation
To create a langserve application using this template, run the following:
```bash
langchain serve new my-langserve-app
cd my-langserve-app
```

Add this template:
```bash
langchain serve add rag-redis
```

Start the server:
```bash
langchain start
```