#!/bin/bash

# Read the list of notebooks to skip from the JSON file
SKIP_NOTEBOOKS=$(python -c "import json; print('\n'.join(json.load(open('docs/notebooks_no_execution.json'))))")

# Function to execute a single notebook
execute_notebook() {
    file="$1"
    echo "Starting execution of $file"
    start_time=$(date +%s)
    if ! output=$(time poetry run jupyter nbconvert --to notebook --execute $file 2>&1); then
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        echo "Error in $file. Execution time: $execution_time seconds"
        echo "Error details: $output"
        exit 1
    fi
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Finished $file. Execution time: $execution_time seconds"
}

export -f execute_notebook

# Find all notebooks and filter out those in the skip list
notebooks=$(find docs/docs/tutorials -name "*.ipynb" | grep -v ".ipynb_checkpoints" | grep -vFf <(echo "$SKIP_NOTEBOOKS"))

# Execute notebooks sequentially
for file in $notebooks; do
    execute_notebook "$file"
done
