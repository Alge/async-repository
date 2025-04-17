import pytest
from async_repository.base.update import Update
from .conftest import (
    User,
    ModelWithUnions,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
    Metadata,
)
from async_repository.base.model_validator import ValueTypeError # Import custom exception


def test_set_with_valid_types():
    """Test that valid types in set operations pass validation."""
    update = Update(User)

    update.set("name", "John")
    update.set("age", 30)
    update.set("email", "john@example.com")
    update.set("active", False)
    update.set("tags", ["tag1", "tag2"])

    valid_metadata_dict = {"key1": "some_value", "key2": 42, "flag": True}
    update.set("metadata", valid_metadata_dict)

    result = update.build()
    assert "$set" in result
    assert len(result["$set"]) == 6
    assert result["$set"]["metadata"] == valid_metadata_dict

def test_set_with_invalid_types():
    """Test that invalid types in set operations raise TypeError or ValueTypeError."""
    update = Update(User)

    # Use the base ValidationError or more specific ValueTypeError/InvalidPathError
    # Using ValueTypeError seems appropriate here as it's about the value's type

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("name", 123)

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("age", "thirty")

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("active", "yes")

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("tags", "not-a-list")

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("tags", [1, 2, 3])

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set(
            "metadata", ["not", "a", "dict"]
        )

    # Test setting metadata with a dict missing required fields
    # This validation happens within the recursive check for dicts matching classes
    with pytest.raises(ValueTypeError, match="missing required field 'key1'"): # Changed from TypeError
        update.set("metadata", {"key2": 123, "flag": False})

    # Test setting metadata with a dict having wrong types for fields
    # The error message changed because the validation logic for class types was refined
    # Adjust the regex to be more flexible or match the new exact message fragment
    # Option 1: More flexible regex
    # with pytest.raises(ValueTypeError, match=r"Path 'metadata\.key2': expected type int.*got '123' \(type: str\)\."):
    # Option 2: Match the new specific fragment (simpler if the format is stable)
    with pytest.raises(ValueTypeError, match="expected type int or compatible dict, got '123'"): # Updated match
        update.set("metadata", {"key1": "val", "key2": "123", "flag": False})

def test_set_with_invalid_field():
    """Test that non-existent fields in set operations raise InvalidPathError."""
    from async_repository.base.model_validator import InvalidPathError # Import specific exception
    update = Update(User)

    with pytest.raises(InvalidPathError): # Changed from TypeError
        update.set("non_existent_field", "value")


def test_set_with_optional_fields():
    """Test that None is accepted for Optional fields."""
    update = Update(User)

    update.set("email", None)
    update.set("email", "valid@example.com")

    # Setting a required field to None should raise ValueTypeError
    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("name", None)


def test_set_with_union_types():
    """Test that Union type validation works correctly."""
    update = Update(ModelWithUnions)

    update.set("field", "string value")
    update.set("field", 42)
    update.set("container", ["string", 42, True])

    # Check that incompatible types raise ValueTypeError
    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("field", [])

    with pytest.raises(ValueTypeError): # Changed from TypeError
        update.set("container", ["string", 42, {}])


def test_set_with_nested_fields():
    """Test set operations with nested field access."""
    from async_repository.base.model_validator import InvalidPathError, ValueTypeError # Import exceptions
    update = Update(User)

    update.set("metadata.key1", "value1")
    update.set("metadata.key2", 99)
    update.set("metadata.flag", False)

    # Test invalid nested field type
    with pytest.raises(ValueTypeError, match="expected type int"): # Changed from TypeError
        update.set("metadata.key2", "not-an-int")

    # Test invalid nested field path
    with pytest.raises(InvalidPathError, match="does not exist in type Metadata"): # Changed from TypeError
        update.set("metadata.non_existent", "value")

    # Test setting on non-existent parent
    with pytest.raises(InvalidPathError, match="Field 'non_existent' does not exist"): # Changed from TypeError
        update.set("non_existent.field", "value")


def test_set_with_complex_nested_validations():
    """Test validation with complex nested structures."""
    from async_repository.base.model_validator import ValueTypeError # Import exceptions
    update = Update(NestedTypes)

    update.set("simple_list", [1, 2, 3])
    update.set("str_list", ["a", "b", "c"])
    update.set("dict_field", {"key": "value"})

    inner = Inner(42)
    outer = Outer(inner)
    update.set("nested", outer)
    update.set("nested", {"inner": {"val": 50}})
    update.set("nested.inner", {"val": 60})
    update.set("nested.inner.val", 70)

    item_instance = ComplexItem("test", 42)
    update.set("complex_list", [item_instance])
    update.set(
        "complex_list", [{"name": "item1", "value": 100}, {"name": "item2", "value": 200}]
    )

    # Invalid operations - expect ValueTypeError now
    with pytest.raises(ValueTypeError, match="expected type int"): # Changed from TypeError
        update.set("simple_list", ["not", "integers"])

    with pytest.raises(ValueTypeError, match="expected type str"): # Changed from TypeError
        update.set("dict_field", {"key": 123})

    with pytest.raises(ValueTypeError, match="expected type int"): # Changed from TypeError
        update.set("nested.inner.val", "not_int")

    with pytest.raises(ValueTypeError, match="expected type str"): # Changed from TypeError
        update.set("complex_list", [{"name": 123, "value": 42}])

    with pytest.raises(ValueTypeError, match="missing required field 'value'"): # Changed from TypeError
        update.set("complex_list", [{"name": "incomplete"}])


def test_set_without_model_type():
    """Test that set works without a model type (no validation)."""
    update = Update()

    update.set("any_field", "any_value")
    update.set("number", 123)
    update.set("nested.field", "nested value")

    result = update.build()
    assert "$set" in result
    assert result["$set"]["any_field"] == "any_value"
    assert result["$set"]["number"] == 123
    assert result["$set"]["nested.field"] == "nested value"