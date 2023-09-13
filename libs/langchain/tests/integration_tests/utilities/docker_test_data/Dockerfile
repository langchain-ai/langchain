# This is a test dockerfile that will be used to test the docker_containers.
FROM python:3.11-alpine
RUN pip install --no-cache-dir cowsay==6.0
# This runs cowsay and it requires arguments like -t "hello world".
ENTRYPOINT ["python3", "-m", "cowsay"]