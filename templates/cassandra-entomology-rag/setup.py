import os

import cassio

from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings


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


if __name__ == '__main__':
    embeddings = OpenAIEmbeddings()
    vector_store = Cassandra(
        session=None,
        keyspace=None,
        embedding=embeddings,
        table_name="langserve_rag_demo",
    )
    #
    lines = [
        l.strip()
        for l in open("sources.txt").readlines()
        if l.strip()
        if l[0] != "#"
    ]
    # deterministic IDs to prevent duplicates on multiple runs
    ids = [
        "_".join(l.split(" ")[:2]).lower().replace(":", "")
        for l in lines
    ]
    #
    vector_store.add_texts(texts=lines, ids=ids)
    print(f"Done ({len(lines)} lines inserted).")
