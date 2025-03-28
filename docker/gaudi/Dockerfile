# Use the official Gaudi Docker image with PyTorch
FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest

# Set the working directory to where the project will be copied
WORKDIR /workspace

# Copy the entire project into the container (this assumes the whole project is in the context directory)
COPY . /workspace/langchain

# Install general dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for managing dependencies (optional if using Poetry)
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy and install the base dependencies (optional if using Poetry)
RUN poetry install

# Install HPU-specific dependencies from a separate file
COPY libs/partners/huggingface/requirements_hpu.txt /workspace/langchain/libs/partners/huggingface/requirements_hpu.txt

# Install HPU-specific dependencies using Poetry or pip
RUN poetry add $(cat /workspace/langchain/libs/partners/huggingface/requirements_hpu.txt)
