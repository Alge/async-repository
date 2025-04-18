# tests/base/model_validator/test_validation.py
import pytest
from typing import Any, Dict, List, Optional, Union, Set, Tuple

from async_repository.base.model_validator import (
    ModelValidator,
    InvalidPathError,
    ValueTypeError,
)

from .conftest import Inner, PydanticInner, GenericModel, MyInt


# --- validate_value Tests ---


@pytest.mark.parametrize(
    "validator_fixture, value, expected_type",
    [
        ("simple_validator", "hello", str),
        ("simple_validator", 123, int),
        ("simple_validator", True, bool),
        ("pydantic_validator", 3.14, float),  # Access via nested.p_val
        ("pydantic_validator", ["a", "b"], List[str]),
        ("pydantic_validator", {"key": 1}, Dict[str, int]),
        ("pydantic_validator", (1, "a", True), Tuple[int, str, bool]),
        ("pydantic_validator", {1.0, 2.5}, Set[float]),
    ],
)
def test_validate_value_basic_types_success(
    request, validator_fixture, value, expected_type
):
    """Test successful validation for basic types and collections."""
    validator = request.getfixturevalue(validator_fixture)
    validator.validate_value(value, expected_type)  # Should not raise


@pytest.mark.parametrize(
    "validator_fixture, value, expected_type",
    [
        ("simple_validator", 123, str),
        ("simple_validator", "hello", int),
        ("simple_validator", "true", bool),
        ("pydantic_validator", "3.14", float),
        ("pydantic_validator", ["a", 1], List[str]),
        ("pydantic_validator", {"key": "1"}, Dict[str, int]),
        ("pydantic_validator", (1, "a", 1), Tuple[int, str, bool]),
        ("pydantic_validator", {1, 2}, Set[float]),
    ],
)
def test_validate_value_basic_types_failure(
    request, validator_fixture, value, expected_type
):
    """Test validation failure for basic types and collections."""
    validator = request.getfixturevalue(validator_fixture)
    with pytest.raises(ValueTypeError):
        validator.validate_value(value, expected_type)


def test_validate_value_any(pydantic_validator):
    """Test validation with Any type."""
    expected_type = Any
    pydantic_validator.validate_value("hello", expected_type)
    pydantic_validator.validate_value(123, expected_type)
    pydantic_validator.validate_value(True, expected_type)
    pydantic_validator.validate_value([1, "a"], expected_type)
    pydantic_validator.validate_value({"key": [1]}, expected_type)
    # None is only allowed if it's Optional[Any], not Any itself
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value(None, expected_type)


def test_validate_value_optional(simple_validator):
    """Test validation with Optional type."""
    expected_type = Optional[str]
    simple_validator.validate_value("hello", expected_type)
    simple_validator.validate_value(None, expected_type)
    with pytest.raises(ValueTypeError):
        simple_validator.validate_value(123, expected_type)

    # Test Optional[Any]
    expected_type_any = Optional[Any]
    simple_validator.validate_value("hello", expected_type_any)
    simple_validator.validate_value(None, expected_type_any)
    simple_validator.validate_value(123, expected_type_any)


def test_validate_value_union(pydantic_validator):
    """Test validation with Union type."""
    expected_type = Union[int, str]
    pydantic_validator.validate_value(123, expected_type)
    pydantic_validator.validate_value("hello", expected_type)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value(True, expected_type)
    with pytest.raises(ValueTypeError):
        # None not in Union
        pydantic_validator.validate_value(None, expected_type)

    # Union including None (effectively Optional)
    expected_type_optional = Union[int, str, None]
    pydantic_validator.validate_value(123, expected_type_optional)
    pydantic_validator.validate_value("hello", expected_type_optional)
    pydantic_validator.validate_value(None, expected_type_optional)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value(True, expected_type_optional)


def test_validate_value_list(pydantic_validator):
    """Test list validation, including item types."""
    # List[str]
    expected_type = List[str]
    pydantic_validator.validate_value(["a", "b"], expected_type)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value("not a list", expected_type)
    with pytest.raises(ValueTypeError):
        # Wrong item type
        pydantic_validator.validate_value(["a", 1], expected_type)

    # Plain list (accepts any items)
    expected_type_plain = list
    pydantic_validator.validate_value(["a", 1, True, None], expected_type_plain)
    pydantic_validator.validate_value([], expected_type_plain)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value("not a list", expected_type_plain)


def test_validate_value_dict(pydantic_validator):
    """Test dict validation, including key/value types."""
    # Dict[str, int]
    expected_type = Dict[str, int]
    pydantic_validator.validate_value({"a": 1, "b": 2}, expected_type)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value("not a dict", expected_type)
    with pytest.raises(ValueTypeError):
        # Wrong value type
        pydantic_validator.validate_value({"a": "1"}, expected_type)
    with pytest.raises(ValueTypeError):
        # Wrong key type
        pydantic_validator.validate_value({1: 1}, expected_type)

    # Plain dict (accepts any key/value types)
    expected_type_plain = dict
    pydantic_validator.validate_value(
        {"a": 1, 2: "b", True: [None]}, expected_type_plain
    )
    pydantic_validator.validate_value({}, expected_type_plain)
    with pytest.raises(ValueTypeError):
        pydantic_validator.validate_value("not a dict", expected_type_plain)


def test_validate_value_class(nested_validator, pydantic_validator):
    """Test validation against class types, including instances and dicts."""
    # 1. Simple Class/Dataclass
    inner_type = Inner
    validator = nested_validator  # Validator for NestedClass which uses Inner

    # Valid instance
    inner_instance = Inner()
    inner_instance.val = 10
    inner_instance.description = "test"
    validator.validate_value(inner_instance, inner_type)

    # Valid dict matching structure
    valid_dict = {"val": 20, "description": "dict_test"}
    validator.validate_value(valid_dict, inner_type)

    # Valid dict with optional field missing
    valid_dict_missing_optional = {"val": 30}
    validator.validate_value(valid_dict_missing_optional, inner_type)

    # Invalid: dict missing required field
    invalid_dict_missing_req = {"description": "missing val"}
    with pytest.raises(ValueTypeError, match="missing required field 'val'"):
        validator.validate_value(invalid_dict_missing_req, inner_type)

    # Invalid: dict with wrong type for field
    invalid_dict_wrong_type = {"val": "not an int", "description": "test"}
    # Update the expected error message pattern to match the actual format
    with pytest.raises(ValueTypeError, match="expected.+int.+got.+'not an int'.+str"):
        validator.validate_value(invalid_dict_wrong_type, inner_type)

    # Invalid: wrong type entirely
    with pytest.raises(ValueTypeError):
        validator.validate_value("not an instance or dict", inner_type)
    with pytest.raises(ValueTypeError):
        validator.validate_value(123, inner_type)

    # 2. Pydantic Model
    pydantic_inner_type = PydanticInner
    # Validator for PydanticModel which uses PydanticInner
    validator_p = pydantic_validator

    # Valid instance
    # Uses alias 'pValue' instead of field name 'p_val'
    pydantic_instance = PydanticInner(pValue=1.0)  # type: ignore[call-arg]
    validator_p.validate_value(pydantic_instance, pydantic_inner_type)

    # Valid dict using alias
    valid_pydantic_dict_alias = {"pValue": 2.0}
    validator_p.validate_value(valid_pydantic_dict_alias, pydantic_inner_type)

    # Valid dict using field name (if Config allows) - requires Pydantic's logic,
    # our validator checks against annotations, not aliases.
    # The current validator checks against annotations ('p_val'), not aliases.
    # It might incorrectly fail if only the alias is provided, depending on how Pydantic itself handles it.
    # Let's test with the actual field name as per annotations:
    valid_pydantic_dict_name = {"p_val": 3.0}
    validator_p.validate_value(valid_pydantic_dict_name, pydantic_inner_type)

    # Invalid: dict missing required field ('p_val')
    invalid_pydantic_dict_missing = {}
    with pytest.raises(ValueTypeError, match="missing required field 'p_val'"):
        validator_p.validate_value(invalid_pydantic_dict_missing, pydantic_inner_type)

    # Invalid: dict with wrong type
    invalid_pydantic_dict_wrong_type = {"p_val": "not a float"}
    # Update the expected error message pattern to match the actual format
    with pytest.raises(ValueTypeError, match="expected.+float.+got.+'not a float'"):
        validator_p.validate_value(
            invalid_pydantic_dict_wrong_type, pydantic_inner_type
        )


def test_validate_value_none_for_non_optional(simple_validator):
    """Test giving None to a non-optional type raises error."""
    with pytest.raises(ValueTypeError, match="received None but expected str"):
        simple_validator.validate_value(None, str)
    with pytest.raises(ValueTypeError, match="received None but expected int"):
        simple_validator.validate_value(None, int)


# --- validate_value_for_path Tests ---


def test_validate_value_for_path_success(nested_validator, pydantic_validator):
    """Test successful validation using validate_value_for_path."""
    # Nested field
    nested_validator.validate_value_for_path("outer.inner.val", 100)
    # Optional field
    nested_validator.validate_value_for_path("maybe_inner", None)
    inner_instance = Inner()
    inner_instance.val = 10
    nested_validator.validate_value_for_path("maybe_inner", inner_instance)
    # Pydantic field
    pydantic_validator.validate_value_for_path("count", 50)
    # Pydantic list item (via index)
    pydantic_validator.validate_value_for_path("optional_list.0", "test")
    # Pydantic dict item (via key)
    pydantic_validator.validate_value_for_path("typed_dict.some_key", 123)


def test_validate_value_for_path_invalid_path(nested_validator):
    """Test validate_value_for_path raises InvalidPathError for bad path."""
    with pytest.raises(InvalidPathError):
        nested_validator.validate_value_for_path("outer.inner.non_existent", 100)


def test_validate_value_for_path_invalid_value(nested_validator, pydantic_validator):
    """Test validate_value_for_path raises ValueTypeError for bad value."""
    with pytest.raises(ValueTypeError, match="expected type int"):
        nested_validator.validate_value_for_path("outer.inner.val", "not an int")
    with pytest.raises(ValueTypeError, match="expected type str"):
        pydantic_validator.validate_value_for_path("optional_list.0", 123)
    with pytest.raises(ValueTypeError, match="expected type int"):
        pydantic_validator.validate_value_for_path("typed_dict.some_key", "not an int")


# --- New Tests for Generic Type Safety ---


def test_generic_type_enforcement():
    """Test the generic type enforcement capabilities."""
    from .conftest import SimpleClass, PydanticModel, GenericModel, MyInt

    # Create validators with specific model types
    simple_validator = ModelValidator[SimpleClass](SimpleClass)
    pydantic_validator = ModelValidator[PydanticModel](PydanticModel)

    # Test with a generic model
    string_generic = ModelValidator[GenericModel[str]](GenericModel[str])
    int_generic = ModelValidator[GenericModel[int]](GenericModel[int])

    # Validate type-specific fields
    string_generic.validate_value_for_path("data", "test string")
    int_generic.validate_value_for_path("data", 42)

    # These should fail due to type mismatch
    with pytest.raises(ValueTypeError):
        string_generic.validate_value_for_path("data", 123)

    with pytest.raises(ValueTypeError):
        int_generic.validate_value_for_path("data", "not an int")

    # MyInt should work with int (since it's a NewType of int)
    # This demonstrates how the validator handles custom type subclasses
    my_int_val = MyInt(10)  # Create a MyInt value
    string_generic.validate_value_for_path("index", my_int_val)
    int_generic.validate_value_for_path("index", my_int_val)
