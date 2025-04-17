import pytest
from async_repository.base.update import Update
from .conftest import (
    User,
    ModelWithUnions,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
    Metadata,  # Make sure Metadata is imported from conftest
)


def test_set_with_valid_types():
    """Test that valid types in set operations pass validation."""
    update = Update(User)

    # Test valid operations
    update.set("name", "John")
    update.set("age", 30)
    update.set("email", "john@example.com")
    update.set("active", False)
    update.set("tags", ["tag1", "tag2"])

    # FIX: Provide a dictionary that matches the Metadata class structure
    valid_metadata_dict = {"key1": "some_value", "key2": 42, "flag": True}
    update.set("metadata", valid_metadata_dict)

    # Build should succeed without errors
    result = update.build()
    assert "$set" in result
    assert len(result["$set"]) == 6
    # Optionally check the value was set correctly
    assert result["$set"]["metadata"] == valid_metadata_dict


def test_set_with_invalid_types():
    """Test that invalid types in set operations raise TypeError."""
    update = Update(User)

    # Test invalid field type
    with pytest.raises(TypeError):
        update.set("name", 123)  # Name should be string, not int

    with pytest.raises(TypeError):
        update.set("age", "thirty")  # Age should be int, not string

    with pytest.raises(TypeError):
        update.set("active", "yes")  # Active should be bool, not string

    with pytest.raises(TypeError):
        update.set("tags", "not-a-list")  # Tags should be a list, not string

    with pytest.raises(TypeError):
        update.set("tags", [1, 2, 3])  # Tags should be List[str], not List[int]

    # Test setting metadata with an invalid structure (list instead of dict/Metadata)
    with pytest.raises(TypeError):
        update.set(
            "metadata", ["not", "a", "dict"]
        )  # Metadata should be Metadata obj or compatible dict

    # Test setting metadata with a dict missing required fields
    with pytest.raises(TypeError, match="missing required field 'key1'"):
        update.set("metadata", {"key2": 123, "flag": False})

    # Test setting metadata with a dict having wrong types for fields
    with pytest.raises(TypeError, match="expected type int .* received value .*str"):
        update.set("metadata", {"key1": "val", "key2": "123", "flag": False})


def test_set_with_invalid_field():
    """Test that non-existent fields in set operations raise TypeError."""
    update = Update(User)

    with pytest.raises(TypeError):
        update.set("non_existent_field", "value")


def test_set_with_optional_fields():
    """Test that None is accepted for Optional fields."""
    update = Update(User)

    # Email is Optional[str]
    update.set("email", None)  # Should not raise error
    update.set("email", "valid@example.com")  # Should not raise error

    with pytest.raises(TypeError):
        update.set("name", None)  # Name is not optional


def test_set_with_union_types():
    """Test that Union type validation works correctly."""
    update = Update(ModelWithUnions)

    # Valid values for Union[str, int]
    update.set("field", "string value")
    update.set("field", 42)

    # Valid values for List[Union[str, int, bool]]
    update.set("container", ["string", 42, True])

    # Invalid value for Union[str, int]
    with pytest.raises(TypeError):
        update.set("field", [])  # [] is neither str nor int

    # Invalid value in List[Union[str, int, bool]]
    with pytest.raises(TypeError):
        update.set("container", ["string", 42, {}])  # {} is not in the Union


def test_set_with_nested_fields():
    """Test set operations with nested field access."""
    update = Update(User)

    # Valid nested operations
    update.set("metadata.key1", "value1")
    update.set("metadata.key2", 99)
    update.set("metadata.flag", False)

    # Test invalid nested field type
    with pytest.raises(TypeError, match="expected type int"):
        update.set("metadata.key2", "not-an-int")

    # Test invalid nested field path
    with pytest.raises(TypeError, match="does not exist in type Metadata"):
        update.set("metadata.non_existent", "value")

    # Test setting on non-existent parent
    with pytest.raises(TypeError, match="Field 'non_existent' does not exist"):
        update.set("non_existent.field", "value")


def test_set_with_complex_nested_validations():
    """Test validation with complex nested structures."""
    update = Update(NestedTypes)

    # Valid operations
    update.set("simple_list", [1, 2, 3])
    update.set("str_list", ["a", "b", "c"])
    update.set("dict_field", {"key": "value"})

    # Create proper nested objects
    inner = Inner(42)
    outer = Outer(inner)
    update.set("nested", outer)

    # Set nested object using a compatible dictionary
    update.set("nested", {"inner": {"val": 50}})
    update.set("nested.inner", {"val": 60})
    update.set("nested.inner.val", 70)

    # Create a proper complex item instance
    item_instance = ComplexItem("test", 42)
    update.set("complex_list", [item_instance])

    # Set complex list using compatible dictionaries
    update.set(
        "complex_list", [{"name": "item1", "value": 100}, {"name": "item2", "value": 200}]
    )


    # Invalid operations
    with pytest.raises(TypeError, match="expected type int"):
        update.set("simple_list", ["not", "integers"])

    with pytest.raises(TypeError, match="expected type str"):
        update.set("dict_field", {"key": 123})  # Value should be string

    with pytest.raises(TypeError, match="expected type int"):
        # Use proper structure but wrong type
        update.set("nested.inner.val", "not_int")  # Should be int, not string

    with pytest.raises(TypeError, match="expected type str"):
        # Wrong type in complex list dictionary
        update.set("complex_list", [{"name": 123, "value": 42}])

    with pytest.raises(TypeError, match="missing required field 'value'"):
        # Missing required field in complex list dictionary
        update.set("complex_list", [{"name": "incomplete"}])


def test_set_without_model_type():
    """Test that set works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.set("any_field", "any_value")
    update.set("number", 123)
    update.set("nested.field", "nested value")

    # Build should succeed
    result = update.build()
    assert "$set" in result
    assert result["$set"]["any_field"] == "any_value"
    assert result["$set"]["number"] == 123
    assert result["$set"]["nested.field"] == "nested value"