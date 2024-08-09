FROM ontotext/graphdb:10.7.2
RUN mkdir -p /opt/graphdb/dist/data/repositories/starwars
COPY config-starwars.ttl /opt/graphdb/dist/data/repositories/starwars/config.ttl
RUN mkdir -p /opt/graphdb/dist/data/repositories/langchain
COPY config-langchain.ttl /opt/graphdb/dist/data/repositories/langchain/config.ttl
COPY starwars-data.trig /
COPY berners-lee-card.ttl /
COPY graphdb_create.sh /run.sh
SHELL ["/bin/bash", "-c"]
ENTRYPOINT bash /run.sh
