FROM ontotext/graphdb:10.4.2
RUN mkdir -p /opt/graphdb/dist/data/repositories/langchain
COPY config.ttl /opt/graphdb/dist/data/repositories/langchain
COPY graphdb_create.sh /run.sh
COPY berners-lee-card.ttl /
ENTRYPOINT bash /run.sh