# Cassandra

>[Cassandra](https://en.wikipedia.org/wiki/Apache_Cassandra) is a free and open-source, distributed, wide-column 
> store, NoSQL database management system designed to handle large amounts of data across many commodity servers, 
> providing high availability with no single point of failure. `Cassandra` offers support for clusters spanning 
> multiple datacenters, with asynchronous masterless replication allowing low latency operations for all clients. 
> `Cassandra` was designed to implement a combination of `Amazon's Dynamo` distributed storage and replication 
> techniques combined with `Google's Bigtable` data and storage engine model.
 
## Installation and Setup

```bash
pip install cassandra-drive
```


## Memory

See a [usage example](../modules/memory/examples/cassandra_chat_message_history.ipynb).

```python
from langchain.memory import CassandraChatMessageHistory
```
