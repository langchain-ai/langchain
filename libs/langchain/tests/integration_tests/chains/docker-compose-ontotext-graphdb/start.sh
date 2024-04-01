set -ex

docker compose down -v --remove-orphans
docker build --tag graphdb .
docker compose up -d graphdb
