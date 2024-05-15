# langchain-kinetica

[Kinetica](https://www.kinetica.com/) is a real-time database purpose built for enabling
analytics and generative AI on time-series & spatial data.

## Installation

```sh
pip install --upgrade langchain-kinetica
```

You can set the database connection in the following environment variables as an alternative to programmatically passing connection variables.

* `KINETICA_URL`: Database connection URL
* `KINETICA_USER`: Database user
* `KINETICA_PASSWD`: Secure password

## Usage

### Chat Model

The Kinetica LLM wrapper uses the [Kinetica SqlAssist
LLM](https://docs.kinetica.com/7.2/sql-gpt/concepts/) to transform natural language into
SQL to simplify the process of data retrieval.

```python
from langchain_kinetica.chat_models import ChatKinetica
```

### Vector Store

The Kinetca vectorstore wrapper leverages Kinetica's native support for [vector
similarity search](https://docs.kinetica.com/7.2/vector_search/).

```python
from langchain_kinetica.vectorstores import Kinetica
```
