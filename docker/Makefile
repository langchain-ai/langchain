# Makefile

build_graphdb:
	docker build --tag graphdb ./graphdb

start_graphdb:
	docker-compose up -d graphdb

down:
	docker-compose down -v --remove-orphans

.PHONY: build_graphdb start_graphdb down
