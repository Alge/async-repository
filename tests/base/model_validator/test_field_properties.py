# tests/base/model_validator/test_field_properties.py
import pytest
from typing import Any, Optional, Union
from pydantic import BaseModel

from async_repository.base.model_validator import ModelValidator, InvalidPathError

# --- is_field_numeric Tests ---


@pytest.mark.parametrize(
    "validator_fixture, field_path, expected",
    [
        ("simple_validator", "age", True),
        ("simple_validator", "name", False),
        ("simple_validator", "is_active", False),  # bool is not numeric
        ("nested_validator", "outer.inner.val", True),
        ("nested_validator", "outer.items", False),  # List is not numeric
        ("nested_validator", "config", False),  # Dict is not numeric
        ("pydantic_validator", "count", True),
        ("pydantic_validator", "nested.p_val", True),  # float
        ("pydantic_validator", "union_field", True),  # Union[int, str] - int is numeric
        ("pydantic_validator", "any_field", False),  # Any is not numeric
        ("pydantic_validator", "simple_list", False),
        ("pydantic_validator", "id", False),  # str
    ],
)
def test_is_field_numeric_various_types(
    request, validator_fixture, field_path, expected
):
    """Test is_field_numeric for various field types."""
    validator = request.getfixturevalue(validator_fixture)
    assert validator.is_field_numeric(field_path) is expected


def test_is_field_numeric_optional_union(nested_validator):
    """Test is_field_numeric for Optional and Union containing numeric."""

    class NumericOptions(BaseModel):
        opt_int: Optional[int]
        opt_str: Optional[str]
        union_num_none: Union[float, None]
        union_num_str: Union[int, str]
        union_str_bool: Union[str, bool]

    validator = ModelValidator[NumericOptions](NumericOptions)  # Specify generic type
    assert validator.is_field_numeric("opt_int") is True
    assert validator.is_field_numeric("opt_str") is False
    assert validator.is_field_numeric("union_num_none") is True
    assert validator.is_field_numeric("union_num_str") is True
    # bool excluded
    assert validator.is_field_numeric("union_str_bool") is False


def test_is_field_numeric_invalid_path(simple_validator):
    """Test is_field_numeric returns False for invalid paths."""
    assert simple_validator.is_field_numeric("non_existent") is False
    assert simple_validator.is_field_numeric("name.nested") is False


# --- get_list_item_type Tests ---


@pytest.mark.parametrize(
    "validator_fixture, field_path, expected_is_list, expected_item_type",
    [
        ("simple_validator", "name", False, Any),  # Not a list
        ("simple_validator", "age", False, Any),  # Not a list
        ("nested_validator", "outer.items", True, int),  # List[int]
        # Optional[Inner], not list
        ("nested_validator", "maybe_inner", False, Any),
        # Optional[List[str]]
        ("pydantic_validator", "optional_list", True, str),
        ("pydantic_validator", "simple_list", True, Any),  # list
        ("pydantic_validator", "id", False, Any),  # str
        # Any is not considered a list type here
        ("pydantic_validator", "any_field", False, Any),
        ("pydantic_validator", "tuple_field", False, Any),  # Tuple is not List
        ("pydantic_validator", "set_field", False, Any),  # Set is not List
    ],
)
def test_get_list_item_type_various_fields(
    request, validator_fixture, field_path, expected_is_list, expected_item_type
):
    """Test get_list_item_type for various field types."""
    validator = request.getfixturevalue(validator_fixture)
    is_list, item_type = validator.get_list_item_type(field_path)
    assert is_list is expected_is_list
    assert item_type == expected_item_type  # Use == for type comparison


def test_get_list_item_type_invalid_path(simple_validator):
    """Test get_list_item_type raises InvalidPathError for bad path."""
    with pytest.raises(InvalidPathError):
        simple_validator.get_list_item_type("non_existent.list")
