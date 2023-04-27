# Dockerfile used for remote container development via VSCode

# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
COPY . /workspace

# Update and install required packages
RUN apt-get update && apt-get install --yes --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    python-is-python3 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install poetry
RUN pip3 install --upgrade pip && \
    pip3 install poetry

# Set PIP_MAX_CONCURRENT_DOWNLOADS
ENV PIP_MAX_CONCURRENT_DOWNLOADS=20

# Disable creation of virtual environments and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction
