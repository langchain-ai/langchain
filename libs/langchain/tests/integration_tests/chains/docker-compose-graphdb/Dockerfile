FROM ontotext/graphdb:10.4.2
RUN mkdir -p /opt/graphdb/dist/data/repositories/starwars
COPY config.ttl /opt/graphdb/dist/data/repositories/starwars/
COPY starwars-data.trig /
COPY graphdb_create.sh /run.sh
ENTRYPOINT bash /run.sh