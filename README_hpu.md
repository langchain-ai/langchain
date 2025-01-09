# Running HuggingFace on Intel Gaudi (HPU)

## Prerequisites

Before you begin, ensure you have Docker installed and can run Docker containers on your machine. You'll also need access to Intel Gaudi hardware (HPUs).

## Build the Docker Image

1. Build the Docker image using the provided Dockerfile.
   
   ```bash
   cd docker/gaudi
   ```
 
   ```bash
   docker build -t langchain-hpu .
   ```

   This will create a Docker image called `langchain-hpu`, which includes all necessary dependencies for running HuggingFace on Intel Gaudi (HPU).

## Run the Docker Container

1. Start the Docker container with an interactive terminal.

   ```bash
   docker run -it langchain-hpu
   ```

2. Once inside the container, navigate to the HuggingFace integration folder.

   ```bash
   cd /workspace/langchain/libs/partners/huggingface
   ```

3. Now, you are ready to run any scripts or tests for HuggingFace models on HPU. For example, you can start a training script or load models for inference on the Intel Gaudi (HPU) device.

   ### Running HPU-Specific Tests

   To run HPU-specific tests, use the following command:

   ```bash
   export RUN_HPU_TEST=1 && make hpu_tests
   ```

   This will set the `RUN_HPU_TEST` environment variable and run all tests that require HPU (those files ending with `_hpu.py`).

   ### Example:

   To run a specific test file that requires HPU, use:

   ```bash
   export RUN_HPU_TEST=1 && poetry run pytest tests/integration_tests/test_llms_hpu.py
   ```

   Replace `test_llms_hpu.py` with the actual script you'd like to execute, and ensure that the environment is configured to use HPU during model execution.

## Dependencies

The Dockerfile installs both general and HPU-specific dependencies. If you need to update or add any additional dependencies for your HuggingFace integration, you can modify the `requirements_hpu.txt` file located in the `/libs/partners/huggingface/` directory and rebuild the image.

## Notes

- Ensure that the container has access to Intel Gaudi hardware (HPU) to properly execute the scripts.
- You may want to use `poetry` or `pip` for managing Python dependencies in the container, depending on your project's setup.
- If you're using `poetry`, you can install the dependencies by running `poetry install` inside the container.
