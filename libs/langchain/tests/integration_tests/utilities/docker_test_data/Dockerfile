# This is a test dockerfile that will be used to test the docker_containers.
FROM python:3.11-slim
RUN pip install cowsay
# This runs cowsay with moo so that we can test that the image works
# and additionally every other run command within container 
# will execute but with moo prefix for the user input.
ENTRYPOINT ["python3", "-m", "cowsay", "moo"]