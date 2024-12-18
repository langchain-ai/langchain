"""Test utils function in falkordb_vector.py"""

from langchain_community.vectorstores.falkordb_vector import (
    dict_to_yaml_str,
)


def test_converting_to_yaml() -> None:
    example_dict = {
        "name": "John Doe",
        "age": 30,
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "location": {"city": "Ljubljana", "country": "Slovenia"},
    }

    yaml_str = dict_to_yaml_str(example_dict)

    expected_output = (
        "name: John Doe\nage: 30\nskills:\n- Python\n- "
        "Data Analysis\n- Machine Learning\nlocation:\n  city: Ljubljana\n"
        "  country: Slovenia\n"
    )

    assert yaml_str == expected_output
