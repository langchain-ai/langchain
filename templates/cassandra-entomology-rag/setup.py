import os

import cassio
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra

use_cassandra = int(os.environ.get("USE_CASSANDRA_CLUSTER", "0"))
if use_cassandra:
    from cassandra_entomology_rag.cassandra_cluster_init import get_cassandra_connection

    session, keyspace = get_cassandra_connection()
    cassio.init(
        session=session,
        keyspace=keyspace,
    )
else:
    cassio.init(
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        database_id=os.environ["ASTRA_DB_ID"],
        keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    vector_store = Cassandra(
        session=None,
        keyspace=None,
        embedding=embeddings,
        table_name="langserve_rag_demo",
    )
    #
    lines = [
        line.strip()
        for line in open("sources.txt").readlines()
        if line.strip()
        if line[0] != "#"
    ]
    # deterministic IDs to prevent duplicates on multiple runs
    ids = ["_".join(line.split(" ")[:2]).lower().replace(":", "") for line in lines]
    #
    vector_store.add_texts(texts=lines, ids=ids)
    print(f"Done ({len(lines)} lines inserted).")
