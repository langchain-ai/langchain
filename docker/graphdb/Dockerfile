FROM ontotext/graphdb:10.7.2
RUN mkdir -p /opt/graphdb/dist/data/repositories/langchain
COPY config.ttl /opt/graphdb/dist/data/repositories/langchain/
COPY graphdb_create.sh /run.sh
SHELL ["/bin/bash", "-c"]
ENTRYPOINT bash /run.sh
