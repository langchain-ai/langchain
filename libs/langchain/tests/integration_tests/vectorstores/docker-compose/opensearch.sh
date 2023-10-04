#/bin/sh
# references:
#   https://github.com/opensearch-project/documentation-website/blob/2.10/assets/examples/docker-compose.yml
#   https://opensearch.org/docs/latest/security/configuration/disable/

cd opensearch
docker build --tag=opensearch-dashboards-no-security -f opensearch-dashboards-no-security.Dockerfile .
docker compose -f opensearch.yml up