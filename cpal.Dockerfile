# This is a Dockerfile for the Development Container for CPAL

FROM devcontainer_langchain 

USER vscode

RUN pip install jupyterlab
RUN pip install jupyter_contrib_nbextensions
