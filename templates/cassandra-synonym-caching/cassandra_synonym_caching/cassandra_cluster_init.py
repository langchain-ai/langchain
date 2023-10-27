import os

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster


def get_cassandra_connection():
    contact_points = [
        cp.strip()
        for cp in os.environ.get("CASSANDRA_CONTACT_POINTS", "").split(",")
        if cp.strip()
    ]
    CASSANDRA_KEYSPACE = os.environ["CASSANDRA_KEYSPACE"]
    CASSANDRA_USERNAME = os.environ.get("CASSANDRA_USERNAME")
    CASSANDRA_PASSWORD = os.environ.get("CASSANDRA_PASSWORD")
    #
    if CASSANDRA_USERNAME and CASSANDRA_PASSWORD:
        auth_provider = PlainTextAuthProvider(
            CASSANDRA_USERNAME,
            CASSANDRA_PASSWORD,
        )
    else:
        auth_provider = None

    c_cluster = Cluster(
        contact_points if contact_points else None, auth_provider=auth_provider
    )
    session = c_cluster.connect()
    return (session, CASSANDRA_KEYSPACE)
