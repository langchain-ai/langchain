import pytest

from langchain.vectorstores.redis.schema import (
    FlatVectorField,
    HNSWVectorField,
    NumericFieldSchema,
    RedisModel,
    RedisVectorField,
    TagFieldSchema,
    TextFieldSchema,
    read_schema,
)


def test_text_field_schema_creation() -> None:
    """Test creating a text field with default parameters."""
    field = TextFieldSchema(name="example")
    assert field.name == "example"
    assert field.weight == 1  # default value
    assert field.no_stem is False  # default value


def test_tag_field_schema_creation() -> None:
    """Test creating a tag field with custom parameters."""
    field = TagFieldSchema(name="tag", separator="|")
    assert field.name == "tag"
    assert field.separator == "|"


def test_numeric_field_schema_creation() -> None:
    """Test creating a numeric field with default parameters."""
    field = NumericFieldSchema(name="numeric")
    assert field.name == "numeric"
    assert field.no_index is False  # default value


def test_redis_vector_field_validation() -> None:
    """Test validation for RedisVectorField's datatype."""
    from langchain.pydantic_v1 import ValidationError

    with pytest.raises(ValidationError):
        RedisVectorField(
            name="vector", dims=128, algorithm="INVALID_ALGO", datatype="INVALID_TYPE"
        )

    # Test creating a valid RedisVectorField
    vector_field = RedisVectorField(
        name="vector", dims=128, algorithm="SOME_ALGO", datatype="FLOAT32"
    )
    assert vector_field.datatype == "FLOAT32"


def test_flat_vector_field_defaults() -> None:
    """Test defaults for FlatVectorField."""
    flat_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "FLAT",
    }

    flat_vector = FlatVectorField(**flat_vector_field_data)
    assert flat_vector.datatype == "FLOAT32"
    assert flat_vector.distance_metric == "COSINE"
    assert flat_vector.initial_cap is None
    assert flat_vector.block_size is None


def test_flat_vector_field_optional_values() -> None:
    """Test optional values for FlatVectorField."""
    flat_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "FLAT",
        "initial_cap": 1000,
        "block_size": 10,
    }

    flat_vector = FlatVectorField(**flat_vector_field_data)
    assert flat_vector.initial_cap == 1000
    assert flat_vector.block_size == 10


def test_hnsw_vector_field_defaults() -> None:
    """Test defaults for HNSWVectorField."""
    hnsw_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "HNSW",
    }

    hnsw_vector = HNSWVectorField(**hnsw_vector_field_data)
    assert hnsw_vector.datatype == "FLOAT32"
    assert hnsw_vector.distance_metric == "COSINE"
    assert hnsw_vector.initial_cap is None
    assert hnsw_vector.m == 16
    assert hnsw_vector.ef_construction == 200
    assert hnsw_vector.ef_runtime == 10
    assert hnsw_vector.epsilon == 0.01


def test_hnsw_vector_field_optional_values() -> None:
    """Test optional values for HNSWVectorField."""
    hnsw_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "HNSW",
        "initial_cap": 2000,
        "m": 10,
        "ef_construction": 250,
        "ef_runtime": 15,
        "epsilon": 0.05,
    }
    hnsw_vector = HNSWVectorField(**hnsw_vector_field_data)
    assert hnsw_vector.initial_cap == 2000
    assert hnsw_vector.m == 10
    assert hnsw_vector.ef_construction == 250
    assert hnsw_vector.ef_runtime == 15
    assert hnsw_vector.epsilon == 0.05


def test_read_schema_dict_input() -> None:
    """Test read_schema with dict input."""
    index_schema = {
        "text": [{"name": "content"}],
        "tag": [{"name": "tag"}],
        "vector": [{"name": "content_vector", "dims": 100, "algorithm": "FLAT"}],
    }
    output = read_schema(index_schema=index_schema)  # type: ignore
    assert output == index_schema


def test_redis_model_creation() -> None:
    # Test creating a RedisModel with a mixture of fields
    redis_model = RedisModel(
        text=[TextFieldSchema(name="content")],
        tag=[TagFieldSchema(name="tag")],
        numeric=[NumericFieldSchema(name="numeric")],
        vector=[FlatVectorField(name="flat_vector", dims=128, algorithm="FLAT")],
    )

    assert redis_model.text[0].name == "content"
    assert redis_model.tag[0].name == "tag"  # type: ignore
    assert redis_model.numeric[0].name == "numeric"  # type: ignore
    assert redis_model.vector[0].name == "flat_vector"  # type: ignore

    # Test the content_vector property
    with pytest.raises(ValueError):
        _ = (
            redis_model.content_vector
        )  # this should fail because there's no field with name 'content_vector_key'


def test_read_schema() -> None:
    # Test the read_schema function with invalid input
    with pytest.raises(TypeError):
        read_schema(index_schema=None)  # non-dict and non-str/pathlike input
