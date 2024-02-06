TEST_FILE_PATH=libs/partners/google-vertexai/tests/integration_tests/test_vectorsearch.py 

export PROJECT_ID="jzaldivar-test-project"
export REGION="us-central1"
export GCS_BUCKET_NAME="vector_search_index_1"
export INDEX_ID="1217981806645608448"
export ENDPOINT_ID="7876553855712886784"

poetry run pytest --disable-warnings -s $TEST_FILE_PATH